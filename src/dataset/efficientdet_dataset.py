# src/dataset/efficientdet_dataset.py
import cv2 
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional

class EfficientDetDataset(Dataset):
    """
    Dataset for EfficientDet model.
    Input: YOLO format annotations.
    Output: Pascal VOC format annotations (absolute pixels for PyTorch).
    """
    def __init__(self, split_txt: Path, img_size: int = 640, transforms: Optional[Callable] = None):
        self.split_txt = split_txt
        self.img_size = img_size
        self.transforms = transforms
        self.img_paths = self._load_imgs(self.split_txt)
    
    def _load_imgs(self, split_txt: Path):
        with open(split_txt, 'r') as f:
            img_paths = [Path(line.strip()) for line in f.readlines() if line.strip()]
        return img_paths

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load YOLO annotations
        label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt'))
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        
                        # Absolute values for width and height  
                        w = abs(w)
                        h = abs(h)
                        if w <= 0.001 or h <= 0.001:
                            continue
                        
                        # Convert YOLO to Albumentations format [x_min, y_min, x_max, y_max]
                        x_min = x_c - (w / 2.0)
                        y_min = y_c - (h / 2.0)
                        x_max = x_c + (w / 2.0)
                        y_max = y_c + (h / 2.0)
                        
                        # Clipping to ensure coordinates are strictly within [0.0, 1.0]
                        x_min = max(0.0, min(1.0, x_min))
                        y_min = max(0.0, min(1.0, y_min))
                        x_max = max(0.0, min(1.0, x_max))
                        y_max = max(0.0, min(1.0, y_max))
                        
                        # Final sanity check
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
        
        # --- APPLY ALBUMENTATIONS TRANSFORMS ---
        # Note: Ensure get_transforms uses format='albumentations' in BboxParams
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']    
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        

        abs_boxes = []
        for box in boxes:
            x_min_n, y_min_n, x_max_n, y_max_n = box
            
            x_min_abs = max(0.0, x_min_n * self.img_size)
            y_min_abs = max(0.0, y_min_n * self.img_size)
            x_max_abs = min(float(self.img_size), x_max_n * self.img_size)
            y_max_abs = min(float(self.img_size), y_max_n * self.img_size)
            
            abs_boxes.append([y_min_abs, x_min_abs, y_max_abs, x_max_abs])

        if len(abs_boxes) == 0:
            target_bbox = torch.zeros((0, 4), dtype=torch.float32)
            target_cls = torch.zeros((0,), dtype=torch.int64)
        else:
            target_bbox = torch.tensor(abs_boxes, dtype=torch.float32)
            target_cls = torch.tensor(labels, dtype=torch.int64) + 1
        target = {
            "bbox": target_bbox,
            "cls": target_cls,
            "img_size": torch.tensor([self.img_size, self.img_size], dtype=torch.float32),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
        }
        return img, target