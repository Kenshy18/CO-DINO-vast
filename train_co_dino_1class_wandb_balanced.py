#!/home/kenke/miniconda3/envs/codino/bin/python
# -*- coding: utf-8 -*-
"""
CO-DINO 1クラス学習スクリプト（WandB統合版・バランス最適化・DDP対応）
公式実装の0.8倍のVRAM使用量を目指したバランス版
DDP実行時は各プロセスが異なるGPUを自動的に使用

使用方法:
# 単一GPU実行
python projects/CO-DETR/train_co_dino_1class_wandb_balanced.py

# DDP実行（2GPU）- 各プロセスが自動的に異なるGPUを使用
torchrun --nproc_per_node=2 projects/CO-DETR/train_co_dino_1class_wandb_balanced.py
torchrun --nproc_per_node=2 projects/CO-DETR/train_co_dino_1class_wandb_balanced.py
"""

import os
import sys
import torch

# CO-DETRモジュールをインポート可能にする
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist, get_rank, get_world_size
from mmdet.utils import setup_cache_size_limit_of_dynamo

# CO-DETRモジュールのインポート
import codetr  # noqa

# カスタムフックのインポート
from custom_hooks import WandbMetricsHook
from simple_visualization_hook import SimpleVisualizationHook
from object_size_analysis_hook_v2 import ObjectSizeAnalysisHookV2

def main():
    # Pytorchのコンパイルキャッシュサイズを設定
    setup_cache_size_limit_of_dynamo()
    
    # ========================================
    # 分散学習の初期化（重要！）
    # ========================================
    # torchrunで実行された場合、環境変数が設定されている
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # MMEngineの分散学習初期化を実行
        # これにより、torch.distributed.init_process_group()が呼ばれ、
        # 各プロセスが異なるGPUを使用し、データが適切に分散される
        init_dist(launcher='pytorch', backend='nccl')
        rank = get_rank()
        world_size = get_world_size()
        print(f"[Process {rank}/{world_size}] Distributed training initialized")
    
    # ========================================
    # 学習設定（ここですべてのパラメータを定義）
    # ========================================
    
    # 基本設定
    num_classes = 1  # 検出するクラス数
    work_dir = './work_dirs/co_dino_1class_wandb_balanced'  # チェックポイント保存ディレクトリ
    
    # WandB設定
    wandb_project = 'co-dino-1class'  # WandBプロジェクト名
    wandb_entity = None  # WandBエンティティ（組織名）。個人の場合はNone
    wandb_run_name = 'co_dino_swin_l_1class_balanced'  # 実行名
    wandb_tags = ['co-dino', '1-class', 'censorship-detection', 'balanced-optimization']  # タグ
    
    # データセット設定
    data_root = '/home/kenke/mmdetection/mmdetection/projects/CO-DETR/censorship_dataset_data1'  # オリジナルデータセットのパス
    train_ann_file = 'annotations/train.json'  # 学習用アノテーションファイル
    val_ann_file = 'annotations/val.json'  # 検証用アノテーションファイル
    train_img_dir = 'train/'  # 学習画像ディレクトリ
    val_img_dir = 'val/'  # 検証画像ディレクトリ
    
    # クラス情報
    metainfo = dict(
        classes=('censorship_target',),  # クラス名
        palette=[(220, 20, 60)]  # 可視化時の色（RGB）
    )
    
    # 学習パラメータ
    batch_size = 1  # バッチサイズ
    num_workers = 2  # データローダーのワーカー数（メモリ削減のため0に）
    max_epochs = 30  # 最大エポック数
    base_lr = 1e-4  # 学習率
    weight_decay = 0.0001  # 重み減衰
    
    # チェックポイント設定
    checkpoint_interval = 3  # チェックポイント保存間隔（エポック）
    max_keep_ckpts = 3  # 保持する最大チェックポイント数
    
    # 可視化設定
    visualization_interval = 10  # 可視化画像の保存間隔（イテレーション）を10に減らして約6枚可視化
    max_vis_samples = 10  # 保存する最大可視化サンプル数
    score_thr = 0.3  # 可視化時のスコア閾値
    
    # 事前学習済みモデル
    backbone_pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
    load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
    
    # ========================================
    # モデル設定
    # ========================================
    
    # 画像サイズとバッチ拡張
    image_size = (1024, 1024)
    batch_augments = [
        dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
    ]
    
    # モデル定義
    model = dict(
        type='CoDETR',
        use_lsj=False,
        eval_module='detr',  # 評価モジュール: 'detr', 'one-stage', 'two-stage'
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=False,
            batch_augments=None
        ),
        
        # バックボーン: Swin-L
        backbone=dict(
            type='SwinTransformer',
            pretrain_img_size=384,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=True,  # gradient checkpointing有効（メモリ削減）
            convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint=backbone_pretrained)
        ),
        
        # ネック
        neck=dict(
            type='ChannelMapper',
            in_channels=[192, 384, 768, 1536],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=5
        ),
        
        # DETR Query Head
        query_head=dict(
            type='CoDINOHead',
            num_query=900,
            num_classes=num_classes,
            in_channels=2048,
            as_two_stage=True,
            dn_cfg=dict(
                label_noise_scale=0.5,
                box_noise_scale=0.4,
                group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)
            ),
            transformer=dict(
                type='CoDinoTransformer',
                with_coord_feat=False,
                num_co_heads=2,
                num_feature_levels=5,
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    with_cp=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0
                        ),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')
                    )
                ),
                decoder=dict(
                    type='DinoTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0
                            ),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                num_levels=5,
                                dropout=0.0
                            ),
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                    )
                )
            ),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                temperature=20,
                normalize=True
            ),
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)
        ),
        
        # RPN Head
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64, 128]
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=12.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=12.0)
        ),
        
        # ROI Head (Faster R-CNN)
        roi_head=[
            dict(
                type='CoStandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32, 64],
                    finest_scale=56
                ),
                bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]
                    ),
                    reg_class_agnostic=False,
                    reg_decoded_bbox=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=12.0
                    ),
                    loss_bbox=dict(type='GIoULoss', loss_weight=120.0)
                )
            )
        ],
        
        # ATSS Head
        bbox_head=[
            dict(
                type='CoATSSHead',
                num_classes=num_classes,
                in_channels=256,
                stacked_convs=1,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[4, 8, 16, 32, 64, 128]
                ),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=12.0
                ),
                loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
                loss_centerness=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=12.0
                )
            )
        ],
        
        # 学習設定
        train_cfg=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]
                )
            ),
            dict(
                rpn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False
                ),
                rpn_proposal=dict(
                    nms_pre=4000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0
                ),
                rcnn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True
                    ),
                    pos_weight=-1,
                    debug=False
                )
            ),
            dict(
                assigner=dict(type='ATSSAssigner', topk=9),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            )
        ],
        
        # テスト設定
        test_cfg=[
            dict(
                max_per_img=300,
                nms=dict(type='soft_nms', iou_threshold=0.8)
            ),
            dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0
                ),
                rcnn=dict(
                    score_thr=0.0,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100
                )
            ),
            dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.6),
                max_per_img=100
            )
        ]
    )
    
    # ========================================
    # データパイプライン設定（バランス最適化版）
    # ========================================
    
    # 学習用データ拡張（バランス版）
    # 公式実装: 最大2048、34種類のスケール、RandomChoice
    # 最適化版: 最大1333、10種類のスケール、RandomChoiceなし
    # バランス版: 最大1333、20種類のスケール、簡略化RandomChoice
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', prob=0.5),
        dict(
            type='RandomChoice',
            transforms=[
                # 選択肢1: 直接リサイズ（70%の確率）
                [
                    dict(
                        type='RandomChoiceResize',
                        # スケールを20種類に増加（公式: 34種類、最適化版: 10種類）
                        # 最大サイズを1333に設定（公式: 2048、最適化版も1333）
                        scales=[
                            (480, 1333), (528, 1333), (576, 1333), (624, 1333),
                            (672, 1333), (720, 1333), (768, 1333), (816, 1333),
                            (864, 1333), (912, 1333), (960, 1333), (1008, 1333),
                            (1056, 1333), (1104, 1333), (1152, 1333), (1200, 1333),
                            (1248, 1333), (1296, 1333), (1333, 1333)
                        ],
                        keep_ratio=True
                    )
                ],
                # 選択肢2: 大きくリサイズしてからクロップ（30%の確率）
                # 公式実装により近いが、計算量を抑えたバージョン
                [
                    dict(
                        type='RandomChoiceResize',
                        scales=[(400, 2666), (500, 2666), (600, 2666)],
                        keep_ratio=True
                    ),
                    dict(
                        type='RandomCrop',
                        crop_type='absolute_range',
                        crop_size=(384, 600),
                        allow_negative_crop=True
                    ),
                    dict(
                        type='RandomChoiceResize',
                        scales=[
                            (480, 1333), (576, 1333), (672, 1333), (768, 1333),
                            (864, 1333), (960, 1333), (1056, 1333), (1152, 1333),
                            (1248, 1333), (1333, 1333)
                        ],
                        keep_ratio=True
                    )
                ]
            ]
        ),
        dict(type='PackDetInputs')
    ]
    
    # テスト用データ処理（バランス版）
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 公式: (2048, 1280)、最適化版も同じ
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
        )
    ]
    
    # ========================================
    # データローダー設定
    # ========================================
    
    train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=False,  # num_workers=0の場合はFalse
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,  # 1回のみ使用（7506枚そのまま）。8000枚相当にしたい場合は2に変更
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file=train_ann_file,
            data_prefix=dict(img=train_img_dir),
            filter_cfg=dict(filter_empty_gt=True),
            pipeline=train_pipeline
        )
    )
)

    val_dataloader = dict(
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file=val_ann_file,
            data_prefix=dict(img=val_img_dir),
            test_mode=False,  # 検証時もGTが必要なのでFalseに設定
            pipeline=test_pipeline
        )
    )
    
    test_dataloader = val_dataloader
    
    # ========================================
    # 評価設定
    # ========================================
    
    val_evaluator = dict(
        type='CocoMetric',
        ann_file=f'{data_root}/{val_ann_file}',
        metric='bbox',
        format_only=False,
        classwise=True  # クラスごとの詳細なメトリクスを有効化
    )
    
    test_evaluator = val_evaluator
    
    # ========================================
    # 最適化設定
    # ========================================
    
    optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
        clip_grad=dict(max_norm=0.1, norm_type=2),
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
        # メモリ削減のための勾配累積（オプション）
        # accumulative_counts=2  # 実質的にバッチサイズ2相当
    )
    
    # 学習率スケジューラー
    param_scheduler = [
        dict(
            type='MultiStepLR',
            begin=0,
            end=max_epochs,
            by_epoch=True,
            milestones=[8,16,24],  # 8エポック目で学習率を0.1倍
            gamma=0.7
        )
    ]
    
    # ========================================
    # ランタイム設定
    # ========================================
    
    # デフォルトスコープ
    default_scope = 'mmdet'
    
    # カスタムフック設定
    custom_hooks = [
        # WandBメトリクス構造化フック
        dict(
            type='WandbMetricsHook',
            interval=10,
            with_step=True,
            log_all_metrics=True,
            priority='NORMAL'  # 優先度を明示的に指定
        ),
        # 可視化フック（最適化版と同じに修正）
        dict(
            type='SimpleVisualizationHook',
            draw=True,
            interval=50,  # 50イテレーションごと
            score_thr=score_thr,
            max_samples=20,
            priority='LOW'  # 他のフックの後に実行
        ),
        # オブジェクトサイズ分析フック（エポック単位版）
        dict(
            type='ObjectSizeAnalysisHookV2',
            interval=100,  # 100イテレーションごとにログ
            priority='NORMAL',
            log_interval_type='iter',  # イテレーション単位でログ
            reset_per_epoch=True  # エポックごとにリセット
        )
    ]
    
    # フック設定
    default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            by_epoch=True,
            interval=checkpoint_interval,
            max_keep_ckpts=max_keep_ckpts,
            save_best='coco/bbox_mAP',  # 最良モデルの保存基準
            rule='greater'
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(
            type='DetVisualizationHook',
            draw=False,  # SimpleVisualizationHookとの競合を避けるため無効化
            interval=visualization_interval,
            score_thr=score_thr,
            show=False,
            wait_time=0.001,
            test_out_dir='visualization'  # 可視化画像の保存先
        )
    )
    
    # 環境設定
    env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl')
    )
    
    # 可視化設定（WandBを含む）
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                tags=wandb_tags,
                config=dict(
                    work_dir=work_dir,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    learning_rate=base_lr,
                    max_epochs=max_epochs,
                    model_name='CO-DINO-Swin-L',
                    dataset=data_root,
                    optimization='balanced',
                    max_scale=1333,
                    num_scales=19,
                    random_choice=True,
                    with_crop_augmentation=True
                )
            )
        )
    ]
    
    visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=vis_backends,
        name='visualizer'
    )
    
    # ログ設定
    log_processor = dict(
        type='LogProcessor',
        window_size=50,
        by_epoch=True
    )
    log_level = 'INFO'
    
    # 学習ループ設定
    train_cfg = dict(
        type='EpochBasedTrainLoop',
        max_epochs=max_epochs,
        val_interval=1
    )
    
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')
    
    # 自動学習率スケーリング（マルチGPU時に使用）
    auto_scale_lr = dict(base_batch_size=16)
    
    # カスタムインポート
    custom_imports = dict(
        imports=[
            'projects.CO-DETR.codetr', 
            'projects.CO-DETR.custom_hooks',
            'projects.CO-DETR.simple_visualization_hook',
            'projects.CO-DETR.object_size_analysis_hook_v2'
        ],
        allow_failed_imports=False
    )
    
    # ========================================
    # 設定を統合
    # ========================================
    
    cfg = Config(
        dict(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            val_evaluator=val_evaluator,
            test_evaluator=test_evaluator,
            optim_wrapper=optim_wrapper,
            param_scheduler=param_scheduler,
            default_scope=default_scope,
            default_hooks=default_hooks,
            custom_hooks=custom_hooks,
            env_cfg=env_cfg,
            vis_backends=vis_backends,
            visualizer=visualizer,
            log_processor=log_processor,
            log_level=log_level,
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
            work_dir=work_dir,
            load_from=load_from,
            resume=False,
            auto_scale_lr=auto_scale_lr,
            custom_imports=custom_imports
        )
    )
    
    # ========================================
    # 学習実行
    # ========================================
    
    print("=" * 60)
    print("CO-DINO 1クラス学習を開始します（バランス最適化版）")
    print("=" * 60)
    print(f"作業ディレクトリ: {work_dir}")
    print(f"データセット: {data_root}")
    print(f"クラス数: {num_classes}")
    print(f"バッチサイズ: {batch_size}")
    print(f"エポック数: {max_epochs}")
    print(f"学習率: {base_lr}")
    print(f"最大画像サイズ: 1333（公式: 2048、最適化版と同じ）")
    print(f"スケール数: 19（公式: 34、最適化版: 10）")
    print(f"RandomChoice: 有効（簡略化版）")
    print(f"推定VRAM使用量: 13-16GB（公式の約65%）")
    print("=" * 60)
    
    # Runnerを作成して学習を実行
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()