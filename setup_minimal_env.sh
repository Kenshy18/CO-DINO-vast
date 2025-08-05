#!/bin/bash

# CO-DINO最小環境セットアップスクリプト

echo "==================================================="
echo "CO-DINO最小環境セットアップを開始します"
echo "==================================================="

# 環境名
ENV_NAME="codino_minimal"

# 既存の環境を削除（存在する場合）
echo "既存の環境を確認中..."
conda env list | grep $ENV_NAME > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "既存の環境 '$ENV_NAME' を削除します..."
    conda deactivate 2>/dev/null
    conda env remove -n $ENV_NAME -y
fi

# 新しいConda環境を作成
echo "新しいConda環境 '$ENV_NAME' を作成中..."
conda create -n $ENV_NAME python=3.9 -y

# 環境をアクティベート
echo "環境をアクティベート中..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# PyTorchのインストール（CUDA 11.8版）
echo "PyTorch (CUDA 11.8) をインストール中..."
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 基本的な依存関係のインストール
echo "基本的な依存関係をインストール中..."
pip install --no-cache-dir \
    numpy \
    matplotlib \
    pycocotools \
    scipy \
    shapely \
    six \
    terminaltables \
    tqdm \
    opencv-python \
    Pillow \
    cython \
    setuptools \
    wheel

# MMEngineとMMCVのインストール
echo "MMEngine と MMCV をインストール中..."
pip install --no-cache-dir mmengine==0.10.3
pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# オプション：WandBとfairscaleのインストール
echo "オプションのパッケージをインストール中..."
pip install --no-cache-dir wandb fairscale

# MMDetectionのインストール（ソースから）
echo "MMDetectionをソースからインストール中..."
cd /home/kenke/mmdetection/mmdetection
pip install -e .

# インストールの確認
echo ""
echo "==================================================="
echo "インストール完了！以下の内容を確認します："
echo "==================================================="
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""
echo "CUDA available:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""
echo "MMEngine version:"
python -c "import mmengine; print(f'MMEngine: {mmengine.__version__}')"
echo ""
echo "MMCV version:"
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
echo ""
echo "MMDetection version:"
python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')"
echo ""

echo "==================================================="
echo "環境セットアップが完了しました！"
echo ""
echo "次のコマンドで環境をアクティベートできます："
echo "  conda activate $ENV_NAME"
echo ""
echo "学習スクリプトを実行するには："
echo "  cd /home/kenke/mmdetection/mmdetection"
echo "  python projects/CO-DETR/co_detr_minimal_repo/train_co_dino_1class_wandb_balanced.py"
echo "===================================================" 