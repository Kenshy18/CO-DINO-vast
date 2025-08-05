"""
データ拡張後のオブジェクトサイズ分布を分析するカスタムフック（エポック単位版）
COCOのサイズ定義に基づいて、small/medium/largeに分類し、WandBに記録する
"""

import numpy as np
import wandb
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ObjectSizeAnalysisHookV2(Hook):
    """データ拡張後のオブジェクトサイズ分布を分析するフック（エポック単位）"""
    
    def __init__(self, 
                 interval=100,
                 priority='NORMAL',
                 log_interval_type='iter',  # 'iter' or 'epoch'
                 reset_per_epoch=True):
        """
        Args:
            interval (int): ログを記録するイテレーション間隔
            priority (str): フックの優先度
            log_interval_type (str): ログ間隔のタイプ（'iter' or 'epoch'）
            reset_per_epoch (bool): エポックごとにカウントをリセットするか
        """
        self.interval = interval
        self.priority = priority
        self.log_interval_type = log_interval_type
        self.reset_per_epoch = reset_per_epoch
        
        # COCOのサイズ定義（ピクセル単位の面積）
        self.size_thresholds = {
            'small': 32 * 32,    # area < 1024
            'medium': 96 * 96    # 1024 <= area < 9216
        }
        
        # エポック内のカウント
        self.epoch_counts = {
            'small': 0,
            'medium': 0,
            'large': 0,
            'total': 0
        }
        
        # 現在のエポック
        self.current_epoch = -1
        
        # エポック内の詳細データ（オプション）
        self.epoch_areas = []
        self.epoch_widths = []
        self.epoch_heights = []
        
        # 最新バッチの情報（リアルタイム表示用）
        self.latest_batch_counts = None
        self.latest_batch_total = 0
    
    def _get_size_category(self, area):
        """面積からサイズカテゴリを判定"""
        if area < self.size_thresholds['small']:
            return 'small'
        elif area < self.size_thresholds['medium']:
            return 'medium'
        else:
            return 'large'
    
    def _reset_epoch_stats(self):
        """エポック統計をリセット"""
        self.epoch_counts = {
            'small': 0,
            'medium': 0,
            'large': 0,
            'total': 0
        }
        self.epoch_areas = []
        self.epoch_widths = []
        self.epoch_heights = []
    
    def _analyze_batch(self, runner, batch_idx, data_batch):
        """バッチ内のオブジェクトサイズを分析"""
        batch_counts = {'small': 0, 'medium': 0, 'large': 0, 'total': 0}
        batch_areas = []
        batch_widths = []
        batch_heights = []
        
        # バッチ内の各画像を処理
        for data_sample in data_batch['data_samples']:
            gt_instances = data_sample.gt_instances
            bboxes = gt_instances.bboxes  # xyxy形式
            
            # トーチテンソルの場合は変換
            if hasattr(bboxes, 'cpu'):
                bboxes = bboxes.cpu().numpy()
            
            for bbox in bboxes:
                # バウンディングボックスのサイズを計算
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # サイズカテゴリを判定
                category = self._get_size_category(area)
                batch_counts[category] += 1
                batch_counts['total'] += 1
                
                # 詳細データを保存
                batch_areas.append(float(area))
                batch_widths.append(float(width))
                batch_heights.append(float(height))
        
        # エポックカウントを更新
        for key in batch_counts:
            self.epoch_counts[key] += batch_counts[key]
        
        # 詳細データを保存
        self.epoch_areas.extend(batch_areas)
        self.epoch_widths.extend(batch_widths)
        self.epoch_heights.extend(batch_heights)
        
        # 最新バッチ情報を保存
        self.latest_batch_counts = batch_counts
        self.latest_batch_total = batch_counts['total']
        
        return batch_counts, batch_areas, batch_widths, batch_heights
    
    def _log_to_wandb(self, runner, force=False):
        """分析結果をWandBにログ"""
        # WandBが初期化されていない場合はスキップ
        if wandb.run is None:
            return
        
        step = runner.iter
        epoch = runner.epoch
        
        # エポック内の分布をログ
        if self.epoch_counts['total'] > 0:
            total = self.epoch_counts['total']
            
            # 現在のエポックのサイズ分布（メインメトリクス）
            epoch_metrics = {
                f'epoch_{epoch}/size_distribution/small': self.epoch_counts['small'] / total,
                f'epoch_{epoch}/size_distribution/medium': self.epoch_counts['medium'] / total,
                f'epoch_{epoch}/size_distribution/large': self.epoch_counts['large'] / total,
                f'epoch_{epoch}/total_objects': total,
            }
            
            # 最新の分布（常に更新）
            current_metrics = {
                'current/small_ratio': self.epoch_counts['small'] / total,
                'current/medium_ratio': self.epoch_counts['medium'] / total,
                'current/large_ratio': self.epoch_counts['large'] / total,
                'current/epoch': epoch,
                'current/total_objects_in_epoch': total,
            }
            
            # 最新バッチの情報
            if self.latest_batch_counts and self.latest_batch_total > 0:
                batch_total = self.latest_batch_total
                current_metrics.update({
                    'latest_batch/small_ratio': self.latest_batch_counts['small'] / batch_total if batch_total > 0 else 0,
                    'latest_batch/medium_ratio': self.latest_batch_counts['medium'] / batch_total if batch_total > 0 else 0,
                    'latest_batch/large_ratio': self.latest_batch_counts['large'] / batch_total if batch_total > 0 else 0,
                    'latest_batch/total_objects': batch_total,
                })
            
            # 統計情報
            if self.epoch_areas:
                stats_metrics = {
                    'current/stats/mean_area': np.mean(self.epoch_areas),
                    'current/stats/median_area': np.median(self.epoch_areas),
                    'current/stats/std_area': np.std(self.epoch_areas),
                    'current/stats/min_area': np.min(self.epoch_areas),
                    'current/stats/max_area': np.max(self.epoch_areas),
                }
                current_metrics.update(stats_metrics)
            
            # メトリクスをログ
            wandb.log({**epoch_metrics, **current_metrics}, step=step)
            
            # カスタムチャート用のテーブル（エポック終了時のみ）
            if force:  # エポック終了時
                # 円グラフ用データ
                pie_data = [
                    ['Small', self.epoch_counts['small']],
                    ['Medium', self.epoch_counts['medium']],
                    ['Large', self.epoch_counts['large']]
                ]
                wandb.log({
                    f'epoch_{epoch}/size_distribution_pie': wandb.Table(
                        columns=['Category', 'Count'],
                        data=pie_data
                    )
                }, step=step)
                
                # ヒストグラム
                if len(self.epoch_areas) > 0:
                    wandb.log({
                        f'epoch_{epoch}/area_histogram': wandb.Histogram(self.epoch_areas[:1000]),
                        f'epoch_{epoch}/width_histogram': wandb.Histogram(self.epoch_widths[:1000]),
                        f'epoch_{epoch}/height_histogram': wandb.Histogram(self.epoch_heights[:1000])
                    }, step=step)
    
    def before_train_epoch(self, runner):
        """エポック開始時の処理"""
        # 新しいエポックの開始を検出
        if runner.epoch != self.current_epoch:
            # 前のエポックの最終ログ（もしあれば）
            if self.current_epoch >= 0 and self.epoch_counts['total'] > 0:
                self._log_to_wandb(runner, force=True)
            
            # エポック番号を更新
            self.current_epoch = runner.epoch
            
            # リセットが有効な場合、統計をリセット
            if self.reset_per_epoch:
                self._reset_epoch_stats()
                runner.logger.info(
                    f"[ObjectSizeAnalysis] Starting epoch {self.current_epoch}, "
                    f"statistics reset."
                )
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """学習イテレーション後に実行"""
        # バッチを分析
        batch_counts, _, _, _ = self._analyze_batch(runner, batch_idx, data_batch)
        
        # 指定間隔でログ
        if self.every_n_train_iters(runner, self.interval):
            self._log_to_wandb(runner)
            
            # コンソールログ
            if runner.iter % (self.interval * 5) == 0:
                total = self.epoch_counts['total']
                if total > 0:
                    runner.logger.info(
                        f"[ObjectSizeAnalysis] Epoch {self.current_epoch} - "
                        f"Total: {total}, "
                        f"Small: {self.epoch_counts['small']} ({self.epoch_counts['small']/total*100:.1f}%), "
                        f"Medium: {self.epoch_counts['medium']} ({self.epoch_counts['medium']/total*100:.1f}%), "
                        f"Large: {self.epoch_counts['large']} ({self.epoch_counts['large']/total*100:.1f}%)"
                    )
    
    def after_train_epoch(self, runner):
        """エポック終了時の処理"""
        if self.epoch_counts['total'] > 0:
            # エポックの最終統計をログ
            self._log_to_wandb(runner, force=True)
            
            # エポックサマリーをコンソールに表示
            total = self.epoch_counts['total']
            runner.logger.info(
                f"\n[ObjectSizeAnalysis] Epoch {self.current_epoch} Summary:\n"
                f"{'='*60}\n"
                f"Total objects: {total}\n"
                f"Small (< 32²): {self.epoch_counts['small']} ({self.epoch_counts['small']/total*100:.1f}%)\n"
                f"Medium (32²-96²): {self.epoch_counts['medium']} ({self.epoch_counts['medium']/total*100:.1f}%)\n"
                f"Large (≥ 96²): {self.epoch_counts['large']} ({self.epoch_counts['large']/total*100:.1f}%)\n"
                f"Mean area: {np.mean(self.epoch_areas):.1f} px² | "
                f"Median: {np.median(self.epoch_areas):.1f} px²\n"
                f"{'='*60}"
            )
    
    def after_train(self, runner):
        """学習終了時の最終処理"""
        # 最終エポックのログを確実に記録
        if self.epoch_counts['total'] > 0:
            self._log_to_wandb(runner, force=True)
        
        runner.logger.info(
            "[ObjectSizeAnalysis] Training completed. "
            "Check WandB for detailed size distribution analysis."
        )