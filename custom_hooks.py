# -*- coding: utf-8 -*-
"""
CO-DINO用カスタムフック
デバッグ用の包括的な可視化とメトリクスロギング機能を提供
"""

import os
import os.path as osp
from typing import Optional, Sequence

import numpy as np
import torch
import cv2
import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
    
try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class WandbMetricsHook(Hook):
    """WandBに構造化されたメトリクスを送信するカスタムフック
    
    CO-DINOの複雑なメトリクスを整理してWandBに送信し、
    ダッシュボードで見やすく表示する。
    """
    
    priority = 'NORMAL'  # フックの優先度を明示的に設定
    
    def __init__(self, 
                 interval: int = 50,
                 with_step: bool = True,
                 log_all_metrics: bool = True):
        self.interval = interval
        self.with_step = with_step
        self.log_all_metrics = log_all_metrics
        
    def after_train_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: dict = None,
                        outputs: dict = None) -> None:
        """学習イテレーション後にメトリクスをWandBに送信"""
        
        if (runner.iter + 1) % self.interval != 0:
            return
            
        # 基本的なメトリクスの取得
        # MMEngine v0.10.x では message_hub を使用
        tag = {}
        if hasattr(runner, 'message_hub'):
            # メッセージハブから最新のログを取得
            for key, log_val in runner.message_hub.log_scalars.items():
                if hasattr(log_val, 'current'):
                    # HistoryBuffer の場合
                    tag[key] = log_val.current()
                elif hasattr(log_val, '__getitem__'):
                    # リストの場合
                    tag[key] = log_val[-1]
        elif outputs is not None and 'log_vars' in outputs:
            # outputs から直接取得
            tag = outputs.get('log_vars', {})
        
        # 構造化されたメトリクスの準備
        metrics = {}
        
        # 全体の損失
        if 'loss' in tag:
            metrics['loss/total'] = tag['loss']
            
        # 主要な損失コンポーネント
        loss_components = ['loss_cls', 'loss_bbox', 'loss_iou']
        for comp in loss_components:
            if comp in tag:
                metrics[f'loss/main/{comp}'] = tag[comp]
                
        # デコーダー層ごとの損失
        for i in range(6):  # 6層のデコーダー
            layer_prefix = f'd{i}'
            for loss_type in ['loss_cls', 'loss_bbox', 'loss_iou']:
                key = f'{layer_prefix}.{loss_type}'
                if key in tag:
                    metrics[f'loss/decoder/layer_{i}/{loss_type}'] = tag[key]
                    
        # Denoising損失
        dn_losses = ['dn_loss_cls', 'dn_loss_bbox', 'dn_loss_iou']
        for dn_loss in dn_losses:
            if dn_loss in tag:
                metrics[f'loss/denoising/{dn_loss}'] = tag[dn_loss]
                
        # エンコーダー損失
        enc_losses = ['enc_loss_cls', 'enc_loss_bbox', 'enc_loss_iou']
        for enc_loss in enc_losses:
            if enc_loss in tag:
                metrics[f'loss/encoder/{enc_loss}'] = tag[enc_loss]
                
        # RPN損失
        if 'loss_rpn_cls' in tag:
            metrics['loss/rpn/cls'] = tag['loss_rpn_cls']
        if 'loss_rpn_bbox' in tag:
            metrics['loss/rpn/bbox'] = tag['loss_rpn_bbox']
            
        # ROI Head損失（Faster R-CNN）
        if 'loss_cls0' in tag:
            metrics['loss/roi_head/cls'] = tag['loss_cls0']
        if 'loss_bbox0' in tag:
            metrics['loss/roi_head/bbox'] = tag['loss_bbox0']
        if 'acc0' in tag:
            metrics['accuracy/roi_head'] = tag['acc0']
            
        # ATSS Head損失
        if 'loss_cls1' in tag:
            metrics['loss/atss_head/cls'] = tag['loss_cls1']
        if 'loss_bbox1' in tag:
            metrics['loss/atss_head/bbox'] = tag['loss_bbox1']
        if 'loss_centerness1' in tag:
            metrics['loss/atss_head/centerness'] = tag['loss_centerness1']
            
        # 補助損失
        for head_idx in range(2):  # 2つの補助ヘッド
            for i in range(6):  # 6層
                aux_cls_key = f'd{i}.loss_cls_aux{head_idx}'
                aux_bbox_key = f'd{i}.loss_bbox_aux{head_idx}'
                aux_iou_key = f'd{i}.loss_iou_aux{head_idx}'
                
                if aux_cls_key in tag:
                    metrics[f'loss/auxiliary/head_{head_idx}/layer_{i}/cls'] = tag[aux_cls_key]
                if aux_bbox_key in tag:
                    metrics[f'loss/auxiliary/head_{head_idx}/layer_{i}/bbox'] = tag[aux_bbox_key]
                if aux_iou_key in tag:
                    metrics[f'loss/auxiliary/head_{head_idx}/layer_{i}/iou'] = tag[aux_iou_key]
                    
        # 学習率とその他のメトリクス
        if 'lr' in tag:
            metrics['train/learning_rate'] = tag['lr']
        if 'time' in tag:
            metrics['train/iter_time'] = tag['time']
        if 'data_time' in tag:
            metrics['train/data_time'] = tag['data_time']
        if 'memory' in tag:
            metrics['train/memory_mb'] = tag['memory']
        if 'grad_norm' in tag:
            metrics['train/grad_norm'] = tag['grad_norm']
            
        # ステップ情報の追加
        if self.with_step:
            metrics['train/epoch'] = runner.epoch + 1
            metrics['train/iter'] = runner.iter + 1
            
        # WandBにログ
        if wandb.run is not None:
            wandb.log(metrics, step=runner.iter)
            
    def after_val_epoch(self,
                       runner: Runner,
                       metrics: dict = None) -> None:
        """検証エポック後にメトリクスをWandBに送信"""
        
        if metrics is None:
            return
            
        wandb_metrics = {}
        
        # COCO評価メトリクス
        if 'coco/bbox_mAP' in metrics:
            wandb_metrics['val/mAP'] = metrics['coco/bbox_mAP']
        if 'coco/bbox_mAP_50' in metrics:
            wandb_metrics['val/mAP_50'] = metrics['coco/bbox_mAP_50']
        if 'coco/bbox_mAP_75' in metrics:
            wandb_metrics['val/mAP_75'] = metrics['coco/bbox_mAP_75']
            
        # サイズ別のAP
        if 'coco/bbox_mAP_s' in metrics:
            wandb_metrics['val/mAP_small'] = metrics['coco/bbox_mAP_s']
        if 'coco/bbox_mAP_m' in metrics:
            wandb_metrics['val/mAP_medium'] = metrics['coco/bbox_mAP_m']
        if 'coco/bbox_mAP_l' in metrics:
            wandb_metrics['val/mAP_large'] = metrics['coco/bbox_mAP_l']
            
        # クラスごとのAP（1クラスの場合）
        for key, value in metrics.items():
            if key.startswith('coco/') and key.endswith('_precision'):
                class_name = key.replace('coco/', '').replace('_precision', '')
                wandb_metrics[f'val/class_AP/{class_name}'] = value
                
        # エポック情報
        wandb_metrics['val/epoch'] = runner.epoch + 1
        
        # WandBにログ
        if wandb.run is not None:
            wandb.log(wandb_metrics, step=runner.iter)


@HOOKS.register_module()
class DebugVisualizationHook(Hook):
    """デバッグ用の詳細な可視化フック
    
    検証時に予測結果とGround Truthを比較した画像を保存し、
    WandBにもアップロードする。
    """
    
    priority = 'NORMAL'  # フックの優先度を明示的に設定
    
    def __init__(self,
                 interval: int = 100,
                 score_thr: float = 0.3,
                 max_samples: int = 10,
                 save_dir: str = 'debug_vis',
                 upload_to_wandb: bool = True):
        self.interval = interval
        self.score_thr = score_thr
        self.max_samples = max_samples
        self.save_dir = save_dir
        self.upload_to_wandb = upload_to_wandb
        self._sample_count = 0
        
    def after_val_iter(self,
                      runner: Runner,
                      batch_idx: int,
                      data_batch: dict,
                      outputs: Sequence[DetDataSample]) -> None:
        """検証イテレーション後に可視化を実行"""
        
        # インターバルチェック
        if batch_idx % self.interval != 0:
            return
            
        # 最大サンプル数のチェック
        if self._sample_count >= self.max_samples:
            return
            
        # 保存ディレクトリの作成
        save_dir = osp.join(runner.work_dir, self.save_dir, f'epoch_{runner.epoch}')
        mkdir_or_exist(save_dir)
        
        # バッチの最初のサンプルを処理
        data_sample = outputs[0]
        img_path = data_sample.img_path
        img_bytes = get(img_path)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        
        # 可視化画像の作成
        visualizer = runner.visualizer
        
        # Ground Truthの描画
        gt_img = img.copy()
        if 'gt_instances' in data_sample:
            gt_instances = data_sample.gt_instances
            visualizer.set_image(gt_img)
            visualizer.draw_bboxes(
                gt_instances.bboxes,
                edge_colors='red',
                line_widths=2
            )
            # dataset_metaを安全に取得（DDP対応）
            dataset_meta = None
            
            # 1. visualizerから試す
            if hasattr(runner.visualizer, 'dataset_meta'):
                dataset_meta = runner.visualizer.dataset_meta
            
            # 2. val_dataloaderから試す（検証中）
            if dataset_meta is None and hasattr(runner, 'val_dataloader') and runner.val_dataloader:
                if hasattr(runner.val_dataloader.dataset, 'metainfo'):
                    dataset_meta = runner.val_dataloader.dataset.metainfo
            
            # 3. train_dataloaderから試す
            if dataset_meta is None and hasattr(runner, 'train_dataloader') and runner.train_dataloader:
                if hasattr(runner.train_dataloader.dataset, 'metainfo'):
                    dataset_meta = runner.train_dataloader.dataset.metainfo
            
            # 4. デフォルト値
            if dataset_meta is None or 'classes' not in dataset_meta:
                dataset_meta = {'classes': ('censorship_target',)}  # 1クラスのデフォルト
            
            # ラベルの安全な処理
            if hasattr(gt_instances, 'labels') and gt_instances.labels is not None and len(gt_instances.labels) > 0:
                class_names = []
                for l in gt_instances.labels:
                    if l is not None:
                        label_idx = int(l)
                        if 0 <= label_idx < len(dataset_meta['classes']):
                            class_names.append(dataset_meta['classes'][label_idx])
                        else:
                            class_names.append(f'class_{label_idx}')
                    else:
                        class_names.append('unknown')
                
                visualizer.draw_texts(
                    class_names,
                    positions=gt_instances.bboxes[:, :2].cpu().numpy() + 5,
                    colors='red',
                    font_sizes=12
                )
            gt_img = visualizer.get_image()
            
        # 予測結果の描画
        pred_img = img.copy()
        if 'pred_instances' in data_sample:
            pred_instances = data_sample.pred_instances
            # スコアフィルタリング
            scores = pred_instances.scores
            inds = scores > self.score_thr
            pred_bboxes = pred_instances.bboxes[inds]
            pred_labels = pred_instances.labels[inds]
            pred_scores = scores[inds]
            
            visualizer.set_image(pred_img)
            visualizer.draw_bboxes(
                pred_bboxes,
                edge_colors='green',
                line_widths=2
            )
            # ラベルとスコアを描画
            texts = []
            for l, s in zip(pred_labels, pred_scores):
                if l is not None:
                    label_idx = int(l)
                    if 0 <= label_idx < len(dataset_meta['classes']):
                        class_name = dataset_meta['classes'][label_idx]
                    else:
                        class_name = f'class_{label_idx}'
                else:
                    class_name = 'unknown'
                texts.append(f"{class_name}: {s:.2f}")
            visualizer.draw_texts(
                texts,
                positions=pred_bboxes[:, :2].cpu().numpy() + 5,
                colors='green',
                font_sizes=12
            )
            pred_img = visualizer.get_image()
            
        # 比較画像の作成（横に並べる）
        h, w = img.shape[:2]
        compare_img = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
        compare_img[:, :w] = img  # 元画像
        compare_img[:, w+10:w*2+10] = gt_img  # GT
        compare_img[:, w*2+20:] = pred_img  # 予測
        
        # テキストの追加
        visualizer.set_image(compare_img)
        visualizer.draw_texts(
            ['Original'],
            positions=np.array([[10, 20]]),
            colors='white',
            font_sizes=16,
            bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}]
        )
        visualizer.draw_texts(
            ['Ground Truth'],
            positions=np.array([[w+20, 20]]),
            colors='white',
            font_sizes=16,
            bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}]
        )
        visualizer.draw_texts(
            ['Prediction'],
            positions=np.array([[w*2+30, 20]]),
            colors='white',
            font_sizes=16,
            bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}]
        )
        compare_img = visualizer.get_image()
        
        # 画像の保存
        img_name = osp.basename(img_path).split('.')[0]
        save_path = osp.join(save_dir, f'{img_name}_compare.jpg')
        mmcv.imwrite(compare_img, save_path)
        
        # 個別画像も保存
        mmcv.imwrite(img, osp.join(save_dir, f'{img_name}_original.jpg'))
        mmcv.imwrite(gt_img, osp.join(save_dir, f'{img_name}_gt.jpg'))
        mmcv.imwrite(pred_img, osp.join(save_dir, f'{img_name}_pred.jpg'))
        
        # WandBへのアップロード
        if self.upload_to_wandb and wandb.run is not None:
            # 比較画像
            wandb.log({
                f'debug_vis/compare/{img_name}': wandb.Image(compare_img),
                f'debug_vis/original/{img_name}': wandb.Image(img),
                f'debug_vis/gt/{img_name}': wandb.Image(gt_img),
                f'debug_vis/pred/{img_name}': wandb.Image(pred_img),
            }, step=runner.iter)
            
            # 統計情報も記録
            if 'gt_instances' in data_sample and 'pred_instances' in data_sample:
                num_gt = len(gt_instances.bboxes)
                num_pred = len(pred_bboxes)
                wandb.log({
                    f'debug_vis/stats/{img_name}/num_gt': num_gt,
                    f'debug_vis/stats/{img_name}/num_pred': num_pred,
                    f'debug_vis/stats/{img_name}/num_pred_filtered': len(pred_bboxes),
                }, step=runner.iter)
                
        self._sample_count += 1
        
    def before_val(self, runner: Runner) -> None:
        """検証開始前にカウンターをリセット"""
        self._sample_count = 0


@HOOKS.register_module()
class EpochSliderVisualizationHook(Hook):
    """エポックごとにスライダー対応の可視化を行うフック"""
    
    priority = 'LOW'  # 他のフックの後に実行
    
    def __init__(self,
                 draw: bool = True,
                 interval: int = 10,
                 score_thr: float = 0.3,
                 max_samples_per_epoch: int = 20,
                 save_dir: str = 'epoch_vis',
                 upload_to_wandb: bool = True,
                 show_class_names: bool = False,
                 line_width: int = 3,
                 font_scale: float = 0.7):
        """
        Args:
            draw: 可視化を行うか
            interval: 可視化間隔（イテレーション）
            score_thr: 予測結果の表示閾値
            max_samples_per_epoch: エポックごとの最大サンプル数
            save_dir: 保存ディレクトリ
            upload_to_wandb: WandBに画像をアップロードするか
            show_class_names: クラス名を表示するか（1クラスなので通常False）
            line_width: バウンディングボックスの線幅
            font_scale: フォントサイズ
        """
        self.draw = draw
        self.interval = interval
        self.score_thr = score_thr
        self.max_samples_per_epoch = max_samples_per_epoch
        self.save_dir = save_dir
        self.upload_to_wandb = upload_to_wandb and (wandb is not None)
        self.show_class_names = show_class_names
        self.line_width = line_width
        self.font_scale = font_scale
        
        # エポックごとの画像を保存
        self._epoch_images = []
        self._epoch_captions = []
        self._sample_count = 0
        
    def before_val_epoch(self, runner: Runner) -> None:
        """検証エポック開始時の処理"""
        self._epoch_images = []
        self._epoch_captions = []
        self._sample_count = 0
        print(f"\n[INFO] Starting validation visualization for epoch {runner.epoch}")
        
    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict = None,
                       outputs: Sequence[DetDataSample] = None) -> None:
        """検証イテレーション後に画像を収集"""
        
        if not self.draw:
            return
        
        # インターバルチェック
        if batch_idx % self.interval != 0:
            return
            
        # 最大サンプル数チェック
        if self._sample_count >= self.max_samples_per_epoch:
            return
        
        try:
            # データサンプルを取得
            if not outputs or len(outputs) == 0:
                return
            data_sample = outputs[0]
            
            # 画像を読み込み
            img_path = getattr(data_sample, 'img_path', None)
            if not img_path or not os.path.exists(img_path):
                return
            
            img = cv2.imread(img_path)
            if img is None:
                return
            
            # 元画像サイズを記録
            orig_h, orig_w = img.shape[:2]
            
            # 比較画像を作成
            comparison_img = self._create_comparison_image(img, data_sample)
            
            # BGR→RGB変換（WandB用）
            comparison_img_rgb = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
            
            # 画像情報
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            # GT数と予測数をカウント
            gt_count = 0
            if hasattr(data_sample, 'gt_instances'):
                gt_instances = data_sample.gt_instances
                if hasattr(gt_instances, 'bboxes'):
                    gt_count = len(gt_instances.bboxes)
                    
            pred_count = 0
            if hasattr(data_sample, 'pred_instances'):
                pred_instances = data_sample.pred_instances
                if hasattr(pred_instances, 'scores') and hasattr(pred_instances, 'bboxes'):
                    scores = pred_instances.scores
                    if torch.is_tensor(scores):
                        scores = scores.cpu().numpy()
                    pred_count = np.sum(scores > self.score_thr)
            
            # キャプションを作成
            caption = (f"Image: {img_id} | "
                      f"Size: {orig_w}x{orig_h} | "
                      f"GT: {gt_count} boxes | "
                      f"Pred: {pred_count} boxes (thr={self.score_thr})")
            
            # エポックの画像リストに追加
            self._epoch_images.append(comparison_img_rgb)
            self._epoch_captions.append(caption)
            self._sample_count += 1
            
            # ローカル保存
            save_dir = os.path.join(runner.work_dir, self.save_dir, f'epoch_{runner.epoch}')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{img_id}_comparison.png')
            cv2.imwrite(save_path, comparison_img)
            
        except Exception as e:
            print(f"[ERROR] Visualization failed at iter {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def after_val_epoch(self,
                       runner: Runner,
                       metrics: Optional[dict] = None) -> None:
        """検証エポック終了時にWandBにアップロード"""
        
        if not self._epoch_images:
            print(f"[WARNING] No images collected for epoch {runner.epoch}")
            return
            
        print(f"[INFO] Collected {len(self._epoch_images)} images for epoch {runner.epoch}")
        
        # WandBにアップロード
        if self.upload_to_wandb and wandb.run is not None:
            try:
                # 画像配列を作成（スライダー対応）
                wandb_images = [
                    wandb.Image(img, caption=caption)
                    for img, caption in zip(self._epoch_images, self._epoch_captions)
                ]
                
                # エポックごとにログ
                # 複数の画像を配列として送ることで、WandBが自動的にスライダーを作成
                wandb.log({
                    f"val/epoch_{runner.epoch}/comparisons": wandb_images,
                    "val/current_epoch": runner.epoch,
                    "val/num_visualized_images": len(self._epoch_images)
                }, step=runner.iter)
                
                # サマリー画像（最初の3枚）も別途ログ
                if len(wandb_images) >= 3:
                    wandb.log({
                        f"val/epoch_{runner.epoch}/sample_1": wandb_images[0],
                        f"val/epoch_{runner.epoch}/sample_2": wandb_images[1],
                        f"val/epoch_{runner.epoch}/sample_3": wandb_images[2],
                    }, step=runner.iter)
                
                print(f"[INFO] Uploaded {len(self._epoch_images)} images to WandB "
                      f"under 'val/epoch_{runner.epoch}/comparisons' with slider support")
                
            except Exception as e:
                print(f"[ERROR] Failed to upload to WandB: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_comparison_image(self, img: np.ndarray, data_sample: DetDataSample) -> np.ndarray:
        """GT（左）と予測（右）の比較画像を作成"""
        
        # Ground Truth用の画像
        gt_img = self._draw_gt_annotations(img.copy(), data_sample)
        
        # 予測結果用の画像
        pred_img = self._draw_pred_annotations(img.copy(), data_sample)
        
        # 左右に並べる
        comparison_img = np.hstack([gt_img, pred_img])
        
        # ヘッダーを追加
        h, w = comparison_img.shape[:2]
        header_height = 60
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header[:] = (40, 40, 40)  # ダークグレー
        
        # タイトルを追加
        cv2.putText(header, "Ground Truth", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(header, "Predictions", (w//2 + 20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # ヘッダーと画像を結合
        comparison_img = np.vstack([header, comparison_img])
        
        return comparison_img
    
    def _draw_gt_annotations(self, img: np.ndarray, data_sample: DetDataSample) -> np.ndarray:
        """Ground Truthアノテーションを描画"""
        
        gt_count = 0
        if hasattr(data_sample, 'gt_instances'):
            gt_instances = data_sample.gt_instances
            if hasattr(gt_instances, 'bboxes') and len(gt_instances.bboxes) > 0:
                bboxes = gt_instances.bboxes
                if torch.is_tensor(bboxes):
                    bboxes = bboxes.cpu().numpy()
                
                # バウンディングボックスを描画（赤色）
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), self.line_width)
                    
                    # ラベル背景
                    label = f"GT {i+1}"
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                    cv2.rectangle(img, (x1, y1 - label_h - 4), 
                                 (x1 + label_w + 4, y1), (0, 0, 255), -1)
                    cv2.putText(img, label, (x1 + 2, y1 - 4), 
                               cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                               (255, 255, 255), 1)
                gt_count = len(bboxes)
        
        # GT数を画像上部に表示
        cv2.putText(img, f"GT Objects: {gt_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return img
    
    def _draw_pred_annotations(self, img: np.ndarray, data_sample: DetDataSample) -> np.ndarray:
        """予測結果のアノテーションを描画"""
        
        pred_count = 0
        if hasattr(data_sample, 'pred_instances'):
            pred_instances = data_sample.pred_instances
            
            if hasattr(pred_instances, 'scores') and hasattr(pred_instances, 'bboxes'):
                scores = pred_instances.scores
                bboxes = pred_instances.bboxes
                
                if torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                if torch.is_tensor(bboxes):
                    bboxes = bboxes.cpu().numpy()
                
                # スコアでソート（降順）
                indices = np.argsort(scores)[::-1]
                
                # バウンディングボックスを描画（緑色）
                for idx, i in enumerate(indices):
                    score = scores[i]
                    if score <= self.score_thr:
                        continue
                        
                    bbox = bboxes[i]
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    
                    # 色の濃さをスコアに応じて調整
                    color_intensity = int(255 * (score - self.score_thr) / (1.0 - self.score_thr))
                    color = (0, color_intensity, 0)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, self.line_width)
                    
                    # ラベル背景
                    label = f"{score:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                    cv2.rectangle(img, (x1, y1 - label_h - 4), 
                                 (x1 + label_w + 4, y1), color, -1)
                    cv2.putText(img, label, (x1 + 2, y1 - 4), 
                               cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                               (255, 255, 255), 1)
                    
                    pred_count += 1
        
        # 予測数と閾値を画像上部に表示
        cv2.putText(img, f"Predictions: {pred_count} (thr={self.score_thr})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img