# CO-DETR Minimal Repository

This is a minimal repository containing only the essential files needed to run CO-DETR single-class detection training with WandB integration.

## Repository Structure

```
co_detr_minimal_repo/
├── README.md
├── requirements.txt
├── setup.sh
├── train_co_dino_1class_wandb_balanced.py
├── codetr/
│   ├── __init__.py
│   ├── codetr.py
│   ├── co_atss_head.py
│   ├── co_dino_head.py
│   ├── co_roi_head.py
│   └── transformer.py
├── custom_hooks.py
├── simple_visualization_hook.py
└── object_size_analysis_hook_v2.py
```

## Requirements

- Python 3.9+
- CUDA 11.8+ compatible GPU (24GB VRAM recommended)
- 32GB+ RAM

## Installation

1. **Clone this repository:**
```bash
git clone <repository-url>
cd co_detr_minimal_repo
```

2. **Run the setup script:**
```bash
./setup.sh
```

Or manually install dependencies:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. **Configure WandB (optional but recommended):**
```bash
wandb login
```

## Dataset Preparation

Prepare your dataset in COCO format:
```
your_dataset/
├── annotations/
│   ├── train.json
│   └── val.json
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
```

## Configuration

Update the following paths in `train_co_dino_1class_wandb_balanced.py`:

```python
# Dataset paths
data_root = '/path/to/your/dataset'
train_ann_file = 'annotations/train.json'
val_ann_file = 'annotations/val.json'
train_img_dir = 'train/'
val_img_dir = 'val/'

# WandB configuration
wandb_project = 'your-project-name'
wandb_entity = 'your-wandb-entity'  # Optional
```

## Training

### Single GPU Training
```bash
python train_co_dino_1class_wandb_balanced.py
```

### Multi-GPU Training (DDP)
```bash
torchrun --nproc_per_node=2 train_co_dino_1class_wandb_balanced.py
```

## Key Features

- **Model**: CO-DETR with Swin-L backbone
- **Single-class detection** optimized for censorship detection
- **WandB integration** for experiment tracking
- **Memory optimization** (~13-16GB VRAM usage)
- **Custom hooks** for visualization and object size analysis

## Model Configuration

- Backbone: Swin-L (pretrained on ImageNet-22K)
- Input size: 1333×1333 (max)
- Batch size: 1 (configurable)
- Learning rate: 1e-4
- Epochs: 30

## Output

Training outputs will be saved to:
```
work_dirs/co_dino_1class_wandb_balanced/
├── epoch_*.pth          # Checkpoints
├── best_coco_bbox_mAP.pth  # Best model
└── vis_data/           # Visualizations
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Enable gradient checkpointing (already enabled)
- Reduce input image size

### Import Errors
- Ensure all files are in the correct directory structure
- Check that MMDetection is properly installed

### WandB Connection Issues
```bash
export WANDB_MODE=offline  # Run in offline mode
```

## License

This minimal repository is derived from MMDetection and CO-DETR projects. Please refer to their respective licenses.

## Acknowledgments

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [CO-DETR](https://github.com/Sense-X/Co-DETR)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)