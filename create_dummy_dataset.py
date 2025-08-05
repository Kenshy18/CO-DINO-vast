#!/usr/bin/env python
"""
Create a minimal dummy dataset for testing CO-DETR training
This creates a small COCO-format dataset with synthetic data
"""

import json
import os
import numpy as np
from PIL import Image

def create_dummy_coco_dataset(output_dir='dummy_dataset', num_images=10):
    """Create a minimal COCO-format dataset for testing"""
    
    # Create directory structure
    os.makedirs(f'{output_dir}/annotations', exist_ok=True)
    os.makedirs(f'{output_dir}/train', exist_ok=True)
    os.makedirs(f'{output_dir}/val', exist_ok=True)
    
    # COCO format structure
    coco_format = {
        "info": {
            "description": "Dummy CO-DETR Dataset",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "censorship_target",
                "supercategory": "object"
            }
        ]
    }
    
    annotation_id = 1
    
    # Create training set
    train_coco = coco_format.copy()
    train_coco["annotations"] = []
    train_coco["images"] = []
    
    for i in range(num_images):
        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))
        
        # Add some random rectangles as objects
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        num_objects = np.random.randint(1, 4)
        for j in range(num_objects):
            x1 = np.random.randint(0, 540)
            y1 = np.random.randint(0, 380)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(50, 100)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Add annotation
            train_coco["annotations"].append({
                "id": annotation_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1
        
        # Save image
        img_filename = f'img_{i:04d}.jpg'
        img.save(f'{output_dir}/train/{img_filename}')
        
        # Add image info
        train_coco["images"].append({
            "id": i + 1,
            "file_name": img_filename,
            "width": 640,
            "height": 480
        })
    
    # Save training annotations
    with open(f'{output_dir}/annotations/train.json', 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    # Create validation set (smaller)
    val_coco = coco_format.copy()
    val_coco["annotations"] = []
    val_coco["images"] = []
    
    for i in range(max(2, num_images // 5)):
        # Create dummy image
        img = Image.new('RGB', (640, 480), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))
        
        # Add one object
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        x1 = np.random.randint(0, 540)
        y1 = np.random.randint(0, 380)
        x2 = x1 + np.random.randint(50, 100)
        y2 = y1 + np.random.randint(50, 100)
        
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # Add annotation
        val_coco["annotations"].append({
            "id": annotation_id,
            "image_id": i + 1,
            "category_id": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "segmentation": [],
            "iscrowd": 0
        })
        annotation_id += 1
        
        # Save image
        img_filename = f'val_{i:04d}.jpg'
        img.save(f'{output_dir}/val/{img_filename}')
        
        # Add image info
        val_coco["images"].append({
            "id": i + 1,
            "file_name": img_filename,
            "width": 640,
            "height": 480
        })
    
    # Save validation annotations
    with open(f'{output_dir}/annotations/val.json', 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"Created dummy dataset in '{output_dir}/'")
    print(f"- Training images: {len(train_coco['images'])}")
    print(f"- Training annotations: {len(train_coco['annotations'])}")
    print(f"- Validation images: {len(val_coco['images'])}")
    print(f"- Validation annotations: {len(val_coco['annotations'])}")

if __name__ == '__main__':
    create_dummy_coco_dataset()