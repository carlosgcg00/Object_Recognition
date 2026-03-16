# src/visualization/visualize_dataset.py

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Optional
from src.visualization.plot_utils import draw_annotations, get_class_color


def plot_image(
    image: np.ndarray, 
    title: str = "Image", 
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Displays BGR image as RGB in Matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
        
    plt.show()
    plt.close(fig)


def visualize_yolo_label(
    image_path: Union[str, Path], 
    label_path: Union[str, Path], 
    class_names: Optional[List[str]] = None
) -> None:
    """Reads image and YOLO txt, denormalizes, and plots."""
    img_path, lbl_path = Path(image_path), Path(label_path)
    
    if not img_path.exists():
        return print(f"Error: {img_path} not found.")
        
    image = cv2.imread(str(img_path))
    h_img, w_img, _ = image.shape
    
    if lbl_path.exists():
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_c, y_c, w_n, h_n = map(float, parts[1:5])
                
                # Denormalize
                w_px, h_px = w_n * w_img, h_n * h_img
                x_min = int((x_c * w_img) - (w_px / 2.0))
                y_min = int((y_c * h_img) - (h_px / 2.0))
                
                color = get_class_color(cls_id)
                name = class_names[cls_id] if class_names else f"Cls {cls_id}"
                
                cv2.rectangle(image, (x_min, y_min), (x_min + int(w_px), y_min + int(h_px)), color, 3)
                cv2.putText(image, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
    plot_image(image, title=f"YOLO: {img_path.name}")


def plot_video_frames_grid(
    video_path: Union[str, Path], 
    start_frame: int, 
    n_frames: int, 
    annotations: Optional[List[Dict]] = None
) -> None:
    """Plots a grid of frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))

    fig, axes = plt.subplots(1, n_frames, figsize=(15, 5))
    if n_frames == 1: axes = [axes]

    for i in range(n_frames):
        curr_id = start_frame + i
        ret, frame = cap.read()
        if not ret: break

        if annotations:
            curr_anns = [a for a in annotations if a['frame_id'] == curr_id]
            frame = draw_annotations(frame, curr_anns)

        axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Frame {curr_id}")
        axes[i].axis('off')
    
    cap.release()
    plt.show()
    plt.close(fig)