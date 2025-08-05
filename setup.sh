#!/bin/bash
# CO-DETR Minimal Repository Setup Script

echo "========================================="
echo "CO-DETR Minimal Repository Setup"
echo "========================================="

# 1. Python環境の確認
echo "Checking Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "Error: Python not found. Please install Python 3.9+"
    exit 1
fi

# 2. 仮想環境の作成（オプション）
echo "Creating virtual environment (optional)..."
# python -m venv venv
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. 依存関係のインストール
echo "Installing dependencies..."
pip install --upgrade pip

# PyTorchのインストール（CUDA 11.8）
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# OpenMIMのインストール
echo "Installing OpenMIM..."
pip install -U openmim

# MMEngineとMMCVのインストール
echo "Installing MMEngine and MMCV..."
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"

# MMDetectionのインストール
echo "Installing MMDetection..."
mim install "mmdet==3.3.0"

# その他の依存関係
echo "Installing other dependencies..."
pip install -r requirements.txt

# 4. インストールの検証
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')"
python -c "import mmengine; print(f'MMEngine: {mmengine.__version__}')"
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo "To run training:"
echo "python train_co_dino_1class_wandb_balanced.py"
echo ""
echo "Note: You need to:"
echo "1. Prepare your dataset in the correct format"
echo "2. Update data paths in the training script"
echo "3. Login to WandB: wandb login"
echo "========================================="