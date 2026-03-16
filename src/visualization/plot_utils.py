# src/visualization/plot_utils.py

import cv2
import numpy as np
from typing import List, Dict, Tuple


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """
    Generates a deterministic pseudo-random color based on the class ID.

    Args:
        class_id: The ID of the class or object.

    Returns:
        Tuple[int, int, int]: An (B, G, R) color tuple for OpenCV.
    """
    np.random.seed(class_id)
    color = np.random.randint(0, 255, size=3).tolist()
    return tuple(color)


def draw_annotations(
    frame: np.ndarray, 
    annotations: List[Dict[str, Any]],
    is_raw: bool = True
) -> np.ndarray:
    """
    Draws bounding boxes and labels on a frame.

    Args:
        frame: The image/frame as a NumPy array (BGR).
        annotations: List of dicts (from analyzer.py or custom).
        is_raw: If True, uses raw pixel coords. If False, assumes YOLO format (not used here).

    Returns:
        np.ndarray: The annotated frame.
    """
    annotated_frame = frame.copy()
    
    for ann in annotations:
        # Compatibility with our analyzer.py keys
        x, y = ann['top_left_x'], ann['top_left_y']
        w, h = ann['width'], ann['height']
        obj_id = ann.get('obj_id', 0)
        cls_id = ann.get('class_id', -1)
        conf = ann.get('conf', 1.0)
        
        color = get_class_color(obj_id if obj_id > 0 else cls_id)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        
        # Label preparation
        label = f"ID:{obj_id} CL:{cls_id} CF:{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale, thick = 0.45, 1
        
        (l_w, l_h), _ = cv2.getTextSize(label, font, f_scale, thick)
        cv2.rectangle(annotated_frame, (x, y - l_h - 10), (x + l_w + 5, y), color, -1)
        cv2.putText(
            annotated_frame, label, (x + 2, y - 7), 
            font, f_scale, (255, 255, 255), thick, cv2.LINE_AA
        )
        
    return annotated_frame