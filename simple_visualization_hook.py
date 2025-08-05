#!/home/kenke/miniconda3/envs/codino/bin/python
# -*- coding: utf-8 -*-
"""
シンプルで確実な可視化フック
ラベルテキストの描画を省略し、バウンディングボックスのみを描画
"""

import os
from typing import Optional, Sequence
import numpy as np
import torch
import cv2
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class SimpleVisualizationHook(Hook):
    """シンプルな可視化フック（バウンディングボックスのみ）"""
    
    priority = 'NORMAL'
    
    def __init__(self,
                 draw: bool = True,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 max_samples: int = 10,
                 save_dir: str = 'simple_vis',
                 upload_to_wandb: bool = True):
        """
        Args:
            draw: 可視化を行うか
            interval: 可視化間隔（イテレーション）
            score_thr: 予測結果の表示閾値
            max_samples: 保存する最大サンプル数
            save_dir: 保存ディレクトリ
            upload_to_wandb: WandBに画像をアップロードするか
        """
        self.draw = draw
        self.interval = interval
        self.score_thr = score_thr
        self.max_samples = max_samples
        self.save_dir = save_dir
        self.upload_to_wandb = upload_to_wandb and (wandb is not None)
        self._sample_count = 0
        
    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict = None,
                       outputs: Sequence[DetDataSample] = None) -> None:
        """検証イテレーション後に可視化を実行"""
        
        if not self.draw or batch_idx % self.interval != 0:
            return
            
        if self._sample_count >= self.max_samples:
            return
        
        # 出力ディレクトリの作成
        save_dir = os.path.join(runner.work_dir, self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
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
            
            # Ground Truth用の画像
            gt_img = img.copy()
            if hasattr(data_sample, 'gt_instances'):
                gt_instances = data_sample.gt_instances
                if hasattr(gt_instances, 'bboxes') and len(gt_instances.bboxes) > 0:
                    # バウンディングボックスを描画（赤色）
                    for bbox in gt_instances.bboxes:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 予測結果用の画像
            pred_img = img.copy()
            if hasattr(data_sample, 'pred_instances'):
                pred_instances = data_sample.pred_instances
                
                # スコアフィルタリング
                if hasattr(pred_instances, 'scores') and hasattr(pred_instances, 'bboxes'):
                    scores = pred_instances.scores
                    bboxes = pred_instances.bboxes
                    
                    # テンソルをnumpyに変換
                    if torch.is_tensor(scores):
                        scores = scores.cpu().numpy()
                    if torch.is_tensor(bboxes):
                        bboxes = bboxes.cpu().numpy()
                    
                    # スコアでフィルタリング
                    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                        if score > self.score_thr:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            # バウンディングボックスを描画（緑色）
                            cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # スコアを表示
                            label = f"{score:.2f}"
                            cv2.putText(pred_img, label, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 比較画像の作成
            comparison_img = np.hstack([gt_img, pred_img])
            
            # タイトルを追加
            h, w = comparison_img.shape[:2]
            cv2.putText(comparison_img, "Ground Truth", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison_img, "Predictions", (w//2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 画像IDを取得
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存
            save_path = os.path.join(save_dir, f'{img_id}_comparison.png')
            cv2.imwrite(save_path, comparison_img)
            print(f"[INFO] Saved visualization: {save_path}")
            
            # WandBにアップロード（BGR→RGB変換が必要）
            if self.upload_to_wandb and wandb.run is not None:
                # wandb.ImageはRGBを期待するので変換
                comparison_img_rgb = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
                wandb.log({
                    f"val/comparison_{self._sample_count}": wandb.Image(
                        comparison_img_rgb, 
                        caption=f"Image: {img_id}"
                    )
                }, step=runner.iter)
            
            self._sample_count += 1
            
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()