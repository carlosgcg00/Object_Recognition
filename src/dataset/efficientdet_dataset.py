import cv2 
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional

class EfficientDetDataset(Dataset):
    """
    Dataset for EfficientDet model with Mosaic Augmentation.
    Input: YOLO format annotations.
    Output: Pascal VOC format annotations (absolute pixels for PyTorch).
    """
    def __init__(self, split_txt: Path, img_size: int = 640, transforms: Optional[Callable] = None, is_train: bool = True, mosaic_prob: float = 0.5):
        """
        Args:
            split_txt (Path): Path to the split text file.
            img_size (int): Size of the images.
            transforms (Optional[Callable]): Albumentations transforms.
            is_train (bool): Whether the dataset is for training.
            mosaic_prob (float): Probability of applying mosaic augmentation.
        """
        
        self.split_txt = split_txt
        self.img_size = img_size
        self.transforms = transforms
        self.is_train = is_train
        # Only apply Mosaic if we are in the training phase
        self.mosaic_prob = mosaic_prob if self.is_train else 0.0
        self.img_paths = self._load_imgs(self.split_txt)
    
    def _load_imgs(self, split_txt: Path):
        with open(split_txt, 'r') as f:
            img_paths = [Path(line.strip()) for line in f.readlines() if line.strip()]
        return img_paths

    def __len__(self):
        return len(self.img_paths)
    
    def load_image_and_boxes(self, idx):
        """Loads a single image and its normalized boxes in [x_min, y_min, x_max, y_max] format."""
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
                        
                        w = abs(w)
                        h = abs(h)
                        if w <= 0.001 or h <= 0.001:
                            continue
                        # From YOLO format (x_center, y_center, width, height) to Pascal VOC format (x_min, y_min, x_max, y_max)
                        x_min = max(0.0, min(1.0, x_c - (w / 2.0)))
                        y_min = max(0.0, min(1.0, y_c - (h / 2.0)))
                        x_max = max(0.0, min(1.0, x_c + (w / 2.0)))
                        y_max = max(0.0, min(1.0, y_c + (h / 2.0)))
                        
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
                            
        return img, boxes, labels

    def load_mosaic(self, index):
        """Loads 2, 3 or 4 images and joins them in a dynamic mosaic."""
        labels_out, boxes_out = [], []
        s = self.img_size
        
        # Base canvas of double size (gray so that empty edges are neutral, if any)
        img_mosaic = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        # Possible templates in format: (x_offset, y_offset, width, height)
        layouts = {
            4: [(0, 0, s, s), (s, 0, s, s), (0, s, s, s), (s, s, s, s)], # Classic 2x2 grid
            3: random.choice([
                [(0, 0, s*2, s), (0, s, s, s), (s, s, s, s)], # 1 large top, 2 small bottom
                [(0, 0, s, s*2), (s, 0, s, s), (s, s, s, s)]  # 1 large left, 2 small right
            ]),
            2: random.choice([
                [(0, 0, s*2, s), (0, s, s*2, s)], # 2 horizontal stacked
                [(0, 0, s, s*2), (s, 0, s, s*2)]  # 2 vertical side by side
            ])
        }
        
        # 1. Randomly choose how many images to use (2, 3 or 4)
        num_imgs = random.choice([2, 3, 4])
        layout = layouts[num_imgs]
        
        # 2. Select random extra indices according to num_imgs
        indices = [index] + random.sample(range(len(self.img_paths)), num_imgs - 1)
        random.shuffle(indices) # Shuffle so that the original image does not always fall in the same place

        # 3. Paste the images in their corresponding quadrant/region
        for i, idx in enumerate(indices):
            img, boxes, labels = self.load_image_and_boxes(idx)
            x_offset, y_offset, w_region, h_region = layout[i]
            
            # Resize the image to fit exactly in the assigned region
            img_resized = cv2.resize(img, (w_region, h_region))
            
            # Paste into the giant mosaic
            img_mosaic[y_offset:y_offset + h_region, x_offset:x_offset + w_region] = img_resized
            
            # Recalculate the coordinates of the boxes [0, 1] with respect to the giant canvas
            for box, label in zip(boxes, labels):
                x_min, y_min, x_max, y_max = box
                
                # Rule of 3: Multiply by the dimension of the region, add the offset, and divide by the total (s*2)
                new_x_min = (x_min * w_region + x_offset) / (s * 2)
                new_y_min = (y_min * h_region + y_offset) / (s * 2)
                new_x_max = (x_max * w_region + x_offset) / (s * 2)
                new_y_max = (y_max * h_region + y_offset) / (s * 2)
                
                boxes_out.append([new_x_min, new_y_min, new_x_max, new_y_max])
                labels_out.append(label)

        # 4. Resize the giant canvas back to the target size (s x s)
        img_mosaic = cv2.resize(img_mosaic, (s, s))
        
        return img_mosaic, boxes_out, labels_out

    def __getitem__(self, idx):
        """Get the image and boxes for the given index."""

        # 1. Mosaic Image or Normal Image
        if random.random() < self.mosaic_prob:
            img, boxes, labels = self.load_mosaic(idx)
        else:
            img, boxes, labels = self.load_image_and_boxes(idx)

        # 2. Albumentations (Color augmentation, cropping, ToTensor, etc)
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']    
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # 3. Final conversion to absolute pixels for EfficientDet [y_min, x_min, y_max, x_max]
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
            target_cls = torch.tensor(labels, dtype=torch.int64) + 1 # +1 because EfficientDet uses 0 for background
            
        target = {
            "bbox": target_bbox,
            "cls": target_cls,
            "img_size": torch.tensor([self.img_size, self.img_size], dtype=torch.float32),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
        }
        
        return img, target