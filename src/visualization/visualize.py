# src/visualization/visualize.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """
    Generates a deterministic pseudo-random color based on the class ID.

    Args:
        class_id (int): The ID of the class.

    Returns:
        Tuple[int, int, int]: An (R, G, B) color tuple.
    """
    np.random.seed(class_id)
    color = np.random.randint(0, 255, size=3).tolist()
    return tuple(color)

def draw_raw_annotations(
    frame: np.ndarray, 
    annotations: List[Dict[str, int]]
) -> np.ndarray:
    """
    Draws bounding boxes and labels on a frame using raw dataset annotations.

    Args:
        frame (np.ndarray): The image/frame as a NumPy array (BGR format).
        annotations (List[Dict[str, int]]): Parsed annotations for this specific frame.

    Returns:
        np.ndarray: The annotated frame.
    """
    annotated_frame = frame.copy()
    
    for ann in annotations:
        x, y = ann['top_left_x'], ann['top_left_y']
        w, h = ann['width'], ann['height']
        obj_id = ann['target_id']
        cls_id = ann['class_id']
        conf = ann['confidence']
        
        color = get_class_color(obj_id) # Unique color per tracked object ID
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare and draw label text
        label = f"ID:{obj_id} CL:{cls_id} CF:{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        back_tl = (x, y - label_h - 10)
        back_br = (x + label_w + 5, y)
        cv2.rectangle(annotated_frame, back_tl, back_br, color, -1)
        
        # Draw text in white
        cv2.putText(annotated_frame, label, (x + 2, y - 7), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated_frame

def plot_image(
    image: np.ndarray, 
    title: str = "Image", 
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Displays an image using Matplotlib and optionally saves it to disk.
    Handles memory cleanup automatically.

    Args:
        image (np.ndarray): The image to display (BGR format, will be converted to RGB).
        title (str): The title of the plot.
        save_path (Optional[Union[str, Path]]): Path to save the figure.
    """
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        
    plt.show()
    plt.close(fig) # CRITICAL: Prevents memory leaks in Jupyter Notebooks

def visualize_yolo_label(
    image_path: Union[str, Path], 
    label_path: Union[str, Path], 
    class_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Reads an image and its corresponding YOLO .txt label, draws them, and displays it.

    Args:
        image_path (Union[str, Path]): Path to the .jpg image.
        label_path (Union[str, Path]): Path to the YOLO .txt annotations.
        class_names (Optional[Dict[int, str]]): Dictionary mapping class IDs to names.
    """
    img_path = Path(image_path)
    lbl_path = Path(label_path)
    
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return
        
    image = cv2.imread(str(img_path))
    height, width, _ = image.shape
    
    if lbl_path.exists():
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls_id = int(parts[0])
            x_c, y_c, w_n, h_n = map(float, parts[1:5])
            
            # Denormalize (convert from 0-1 scale back to pixels)
            w_px = w_n * width
            h_px = h_n * height
            x_min = int((x_c * width) - (w_px / 2.0))
            y_min = int((y_c * height) - (h_px / 2.0))
            
            color = get_class_color(cls_id) # Unique color per class
            name = class_names[cls_id] if class_names and cls_id in class_names else f"Class {cls_id}"
            
            cv2.rectangle(image, (x_min, y_min), (x_min + int(w_px), y_min + int(h_px)), color, 3)
            cv2.putText(image, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    else:
        print(f"Warning: No labels found at {lbl_path}. Showing raw image.")

    plot_image(image, title=f"YOLO Annotations: {img_path.name}")