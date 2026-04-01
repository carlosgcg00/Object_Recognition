# src/visualization/visualize.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import torch


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
    Generates a unique color for each bounding box to simulate instance tracking.

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
            
        # We iterate with an index (i) to use it as a pseudo-ID for colors
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls_id = int(parts[0])
            x_c, y_c, w_n, h_n = map(float, parts[1:5])
            
            # Denormalize
            w_px = w_n * width
            h_px = h_n * height
            x_min = int((x_c * width) - (w_px / 2.0))
            y_min = int((y_c * height) - (h_px / 2.0))
            
            # 1. Use the line index (i) to generate a unique color for each object
            color = get_class_color(cls_id + i * 10) 
            
            name = class_names[cls_id] if class_names and cls_id in class_names else f"CL:{cls_id}"
            label = f"{name}" # We add ID:? because YOLO doesn't save tracking IDs
            
            # 2. Draw bounding box (thicker for better visibility)
            cv2.rectangle(image, (x_min, y_min), (x_min + int(w_px), y_min + int(h_px)), color, 3)
            
            # 3. Enhanced text drawing with background for better readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 # Increased size
            thickness = 2    # Increased thickness
            
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            back_tl = (x_min, y_min - label_h - 10)
            back_br = (x_min + label_w + 5, y_min)
            cv2.rectangle(image, back_tl, back_br, color, -1)
            
            # White text over colored background
            cv2.putText(image, label, (x_min + 2, y_min - 7), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    else:
        print(f"Warning: No labels found at {lbl_path}. Showing raw image.")

    # We use our plot_image function to ensure it displays correctly and cleans up memory
    plot_image(image, title=f"YOLO Annotations: {img_path.name}")


def plot_video_frames_grid(
    video_path: Union[str, Path], 
    start_frame: int, 
    n_frames: int, 
    annotations: Optional[List[Dict[str, int]]] = None, 
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Displays a grid of N consecutive frames from a video, optionally with annotations.

    Args:
        video_path (Union[str, Path]): Path to the video file.
        start_frame (int): The 1-based index of the first frame to display.
        n_frames (int): Number of consecutive frames to plot.
        annotations (Optional[List[Dict[str, int]]]): Parsed dataset annotations.
        save_path (Optional[Union[str, Path]]): Path to save the resulting figure.
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {path_str}")
        return

    # OpenCV uses 0-based indexing for CAP_PROP_POS_FRAMES
    # Assuming start_frame is 1-based (like in the Ground Truth)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))

    fig, axes = plt.subplots(1, n_frames, figsize=(5 * n_frames, 5), dpi=150)
    
    # Handle the case where n_frames is 1 (axes is not an array)
    if n_frames == 1:
        axes = [axes]

    for i in range(n_frames):
        current_frame_id = start_frame + i
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Reached end of video at frame {current_frame_id - 1}")
            break

        if annotations:
            # Filter annotations for the current frame
            current_anns = [ann for ann in annotations if ann['frame_id'] == current_frame_id]
            frame = draw_raw_annotations(frame, current_anns)

        # Convert to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(frame_rgb)
        axes[i].set_title(f"Frame {current_frame_id}")
        axes[i].axis('off')
    
    cap.release()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        
    plt.show()
    plt.close(fig) # Liberar memoria


def export_annotated_video(
    video_path: Union[str, Path], 
    annotations: List[Dict[str, int]], 
    output_path: Union[str, Path]
) -> None:
    """
    Processes a full video, draws annotations frame by frame, and exports to a new file.

    Args:
        video_path (Union[str, Path]): Path to the raw video.
        annotations (List[Dict[str, int]]): Parsed dataset annotations.
        output_path (Union[str, Path]): Destination path for the .mp4 file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path.name}")
        return

    # Extract original video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure the VideoWriter (using mp4v codec for standard .mp4 compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_id = 1
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Filter annotations for this specific frame
            current_anns = [ann for ann in annotations if ann['frame_id'] == frame_id]
            
            # Draw using our unified function
            annotated_frame = draw_raw_annotations(frame, current_anns)
            
            writer.write(annotated_frame)
            frame_id += 1
            
    finally:
        # Ensures resources are released even if an error occurs mid-process
        cap.release()
        writer.release()
        print(f"Success: Annotated video saved to {output_path}")


def save_evaluation_grid(img_tensor, gt_dict, pred_tensor, class_names, out_path, score_thresh=0.3):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_np = np.ascontiguousarray(img_np)
    
    GT_COLOR = (0, 255, 0)   
    PRED_COLOR = (0, 0, 255) 
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    gt_boxes = gt_dict['bbox'].cpu().numpy()
    gt_labels = gt_dict['cls'].cpu().numpy()
    
    for box, label_id in zip(gt_boxes, gt_labels):
        y_min, x_min, y_max, x_max = map(int, box) 
        name = class_names.get(int(label_id), f"ID:{label_id}")
        
        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), GT_COLOR, 2)
        label_txt = f"GT:{name}"
        (w, h), _ = cv2.getTextSize(label_txt, FONT, 0.4, 1)
        cv2.rectangle(img_np, (x_min, y_min - h - 5), (x_min + w, y_min), GT_COLOR, -1)
        cv2.putText(img_np, label_txt, (x_min, y_min - 5), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    if pred_tensor is not None and len(pred_tensor) > 0:
        preds = pred_tensor.cpu().numpy()
        valid_preds = preds[preds[:, 4] > score_thresh]
        
        for pred in valid_preds:
            x_min, y_min, x_max, y_max, score, label_id = pred 
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            name = class_names.get(int(label_id), f"ID:{label_id}")

            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), PRED_COLOR, 2)
            pred_txt = f"PR:{name} {score:.2f}"
            (w, h), _ = cv2.getTextSize(pred_txt, FONT, 0.4, 1)
            cv2.rectangle(img_np, (x_min, y_max), (x_min + w, y_max + h + 5), PRED_COLOR, -1)
            cv2.putText(img_np, pred_txt, (x_min, y_max + h + 2), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img_np)

def plot_training_curves(history: dict, output_dir: Path, epochs_TL: int, class_names: dict):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    def add_transition_line(ax):
        ax.axvline(x=epochs_TL, color='gray', linestyle='--', linewidth=2, label='FT Start')

    # 1. Loss Breakdown
    axs[0, 0].plot(epochs, history['train_loss'], label='Total Loss', color='black', linewidth=2)
    axs[0, 0].plot(epochs, history['train_class_loss'], label='Class Loss', color='black', linestyle='--')
    axs[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axs[0, 0].plot(epochs, history['val_class_loss'], label='Val Class Loss', color='red', linestyle='--')
    add_transition_line(axs[0, 0])
    axs[0, 0].set_title('Loss Components Breakdown')
    axs[0, 0].set_ylabel('Loss Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Global mAP
    axs[0, 1].plot(epochs, history['val_map50'], label='mAP@50', color='green', marker='o')
    axs[0, 1].plot(epochs, history['val_map75'], label='mAP@75', color='darkgreen', marker='x')
    add_transition_line(axs[0, 1])
    axs[0, 1].set_title('Global Detection Performance')
    axs[0, 1].set_ylabel('mAP Score')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rate
    axs[1, 0].plot(epochs, history['lr'], label='LR', color='orange', linewidth=2)
    add_transition_line(axs[1, 0])
    axs[1, 0].set_title('Learning Rate Schedule')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Learning Rate')
    axs[1, 0].set_yscale('log') 
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Per-Class mAP@50
    colors = ['purple', 'cyan', 'magenta', 'brown', 'pink']
    for idx, (c_id, map_list) in enumerate(history['val_map50_per_class'].items()):
        c_name = class_names[c_id]
        c_color = colors[idx % len(colors)]
        axs[1, 1].plot(epochs, map_list, label=f'{c_name.capitalize()} (ID:{c_id})', color=c_color, marker='s', markersize=4)
        
    add_transition_line(axs[1, 1])
    axs[1, 1].set_title('mAP@50 per Class')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('mAP@50 Score')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dashboard.png', dpi=200)
    print(f"✅ Training dashboard saved to {output_dir / 'training_dashboard.png'}")
    plt.close(fig)