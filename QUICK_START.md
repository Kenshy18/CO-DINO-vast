# CO-DETR Minimal Repository - Quick Start Guide

## 1. 最速セットアップ（5分）

```bash
# 1. セットアップスクリプトを実行
./setup.sh

# 2. ダミーデータセットを作成（テスト用）
python create_dummy_dataset.py

# 3. データパスを更新
sed -i "s|/home/kenke/mmdetection/mmdetection/projects/CO-DETR/censorship_dataset_data1|./dummy_dataset|g" train_co_dino_1class_wandb_balanced.py
```

## 2. 実際のデータセットで使用する場合

### データセット準備
```bash
# COCOフォーマットのデータセットを準備
your_dataset/
├── annotations/
│   ├── train.json  # COCO形式のアノテーション
│   └── val.json
├── train/         # 学習画像
└── val/           # 検証画像
```

### 設定変更（train_co_dino_1class_wandb_balanced.py）
```python
# 68行目付近
data_root = '/path/to/your_dataset'  # あなたのデータセットパスに変更
```

## 3. 学習実行

### WandBなしで実行（オフライン）
```bash
export WANDB_MODE=offline
python train_co_dino_1class_wandb_balanced.py
```

### WandBありで実行
```bash
wandb login  # 初回のみ
python train_co_dino_1class_wandb_balanced.py
```

## 4. Vast.aiでの使用

```bash
# 1. アーカイブをアップロード
scp -P <PORT> co_detr_minimal_repo.tar.gz root@<VAST_AI_IP>:/workspace/

# 2. Vast.aiインスタンスで解凍
tar -xzf co_detr_minimal_repo.tar.gz
cd co_detr_minimal_repo

# 3. セットアップと実行
./setup.sh
python train_co_dino_1class_wandb_balanced.py
```

## 5. メモリ不足の場合

```python
# train_co_dino_1class_wandb_balanced.pyの以下を変更:
batch_size = 1  # すでに1なので変更不要
num_workers = 0  # 2から0に減らす（73行目）

# 画像サイズを小さくする（512行目付近）
scales=[
    (480, 800), (512, 800), (544, 800), (576, 800),  # 1333を800に
    # ...
]
```

## 6. トラブルシューティング

### ImportError: No module named 'codetr'
```bash
# Pythonパスを確認
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### CUDA out of memory
- 上記のメモリ削減設定を適用
- GPUメモリが24GB未満の場合は画像サイズをさらに小さく

### WandB接続エラー
```bash
export WANDB_MODE=offline
```

## 必要最小限のGPU要件
- VRAM: 16GB以上（推奨24GB）
- CUDA: 11.8以上
- Driver: 470.57以上