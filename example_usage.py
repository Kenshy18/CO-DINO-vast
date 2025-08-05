#!/usr/bin/env python
"""
Example usage script for CO-DETR training
This shows how to use the training script with custom configurations
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main training function
from train_co_dino_1class_wandb_balanced import main

# Custom configuration example
def run_custom_training():
    """Example of running training with custom settings"""
    
    # Set environment variables for custom paths
    os.environ['_CO_DETR_DATA_ROOT'] = '/path/to/your/dataset'
    os.environ['_CO_DETR_WORK_DIR'] = './work_dirs/custom_experiment'
    os.environ['_WANDB_PROJECT'] = 'my-codetr-project'
    os.environ['_WANDB_RUN_NAME'] = 'custom-run-001'
    
    # You can also modify other settings by patching the main function
    # or by creating a modified version of the training script
    
    print("Starting custom CO-DETR training...")
    print(f"Data root: {os.environ.get('_CO_DETR_DATA_ROOT')}")
    print(f"Work dir: {os.environ.get('_CO_DETR_WORK_DIR')}")
    print(f"WandB project: {os.environ.get('_WANDB_PROJECT')}")
    
    # Run training
    main()

if __name__ == '__main__':
    # Basic usage - just run the original training
    # main()
    
    # Custom usage with modified paths
    run_custom_training()