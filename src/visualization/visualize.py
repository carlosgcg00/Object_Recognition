# src/visualization/visualize.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import torch
from tqdm import tqdm
import pandas as pd


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
        font_scale =1
        thickness = 2
        
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
    plt.close(fig)


def plot_class_samples(scanned_data, class_name, frame_id=100, save_path = None):
    """
    Plots a grid showing one specific frame from every video of a given class.
    Includes the video name and its resolution in the title.
    
    Args: 
    - scanned_data: The full dataset information as returned by the scanning function.
    - class_name: The name of the class to filter videos by (e.g., 'car').
    - frame_id: The specific frame number to extract from each video (1-based index).
    - save_path: Optional path to save the resulting figure as a PDF.
    
    """
    # Filtrar solo los vídeos de la clase especificada
    videos_list = scanned_data['data'].get(class_name, [])
    n_videos = len(videos_list)
    
    if n_videos == 0:
        print(f"No hay videos para la clase {class_name}")
        return

    # Calcular filas y columnas para el subplot (ej. 2 filas de 3 columnas para 6 vídeos)
    cols = 3
    rows = (n_videos + cols - 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), dpi=100)
    axes = axes.flatten() # Aplanar para iterar fácilmente

    for i, item in enumerate(videos_list):
        video_path = str(item['video'])
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Extraer resolución
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Leer el frame específico
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if ret:
                # Convertir a RGB para Matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                axes[i].imshow(frame_rgb)
                # Título con nombre del vídeo y resolución
                axes[i].set_title(f"{item['video'].stem}\nResolución: {w}x{h}", fontsize=12)
            else:
                axes[i].set_title(f"Error reading frame\n{item['video'].stem}")
                
            cap.release()
            
        axes[i].axis('off')

    # Ocultar los ejes sobrantes si n_videos no es múltiplo de cols
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), format = 'pdf', dpi=150, bbox_inches='tight')
        
    plt.show()
    plt.close(fig)


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

    fig, axes = plt.subplots(1, n_frames, figsize=(10 * n_frames, 10), dpi=150)
    
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
        plt.savefig(str(save_path), format = 'pdf', dpi=150, bbox_inches='tight')
        
    plt.show()
    plt.close(fig) 


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
    """
    Saves a grid visualization comparing ground truth and predicted bounding boxes.

    Args:
        img_tensor (torch.Tensor): Input image tensor (C, H, W).
        gt_dict (dict): Ground truth annotations with 'bbox' and 'cls' keys.
        pred_tensor (torch.Tensor): Predicted annotations with 'bbox' and 'cls' keys.
        class_names (dict): Mapping of class IDs to names.
        out_path (str): Path to save the output image.
        score_thresh (float): Confidence threshold for predictions.
    Returns:
        None
    """
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
    """
    Plots training and validation loss curves.

    Args:
        history (dict): Dictionary containing training history.
        output_dir (Path): Directory to save the plots.
        epochs_TL (int): Number of epochs for transfer learning.
        class_names (dict): Mapping of class IDs to names.
    Returns:
        None
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=200)
    
    def add_transition_line(ax):
        ax.axvline(x=epochs_TL, color='gray', linestyle='--', linewidth=2, label='FT Start')

    # 1. Loss Breakdown
    axs[0, 0].plot(epochs, history['train_loss'], label='Total Loss', color='black', linewidth=2)
    axs[0, 0].plot(epochs, history['train_class_loss'], label='Class Loss', color='black', linestyle='--')
    axs[0, 0].plot(epochs, history['train_box_loss'], label='Box Loss', color='black', linestyle='-.')
    axs[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axs[0, 0].plot(epochs, history['val_class_loss'], label='Val Class Loss', color='red', linestyle='--')
    axs[0, 0].plot(epochs, history['val_box_loss'], label='Val Box Loss', color='red', linestyle='-.')
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
    
def predict_video_with_model(
    model: torch.nn.Module,
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    class_names: Dict[int, str],
    img_size: int = 512,
    score_thresh: float = 0.50,
    device: str = 'cuda'
) -> None:
    """
    Processes a video frame by frame using the trained model, draws the predictions
    and exports a new video in .mp4 format
    
    Args:
        model (torch.nn.Module): The trained model.
        video_path (Union[str, Path]): Path to the input video.
        output_path (Union[str, Path]): Path to save the output video.
        class_names (Dict[int, str]): Mapping of class IDs to names.
        img_size (int): Size of the input images.
        score_thresh (float): Confidence threshold for predictions.
        device (str): Device to use for inference.
    Returns:
        None
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Ensure the model is in inference mode
    model.eval()
    model.switch_to_predict()
    model.to(device)

    # 2. Open the original video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el vídeo {video_path}")
        return

    # Extract properties for the VideoWriter
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))

    # Normalization constants (same as Albumentations)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print(f"🎥 Processing video: {video_path.name}")
    
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # --- PREPROCESSING ---
        # 1. BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. Resize to the size expected by the model (e.g., 512x512)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        # 3. Scale to [0, 1] and normalize
        img_normalized = (img_resized / 255.0 - mean) / std
        # 4. Convert to Tensor [C, H, W], add batch [1, C, H, W] and move to GPU
        img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # --- INFERENCE ---
        with torch.no_grad():
            detections = model(img_tensor)[0] # Get the first element of the batch

        # --- POSTPROCESSING AND DRAWING ---
        if detections is not None and len(detections) > 0:
            preds = detections.cpu().numpy()
            # Filter by confidence threshold
            valid_preds = preds[preds[:, 4] > score_thresh]

            for pred in valid_preds:
                x_min_n, y_min_n, x_max_n, y_max_n, score, cls_id = pred
                
                # The coordinates are referenced to the size img_size (512x512)
                # They must be rescaled to the actual size of the video (orig_w, orig_h)
                x_min = int((x_min_n / img_size) * orig_w)
                y_min = int((y_min_n / img_size) * orig_h)
                x_max = int((x_max_n / img_size) * orig_w)
                y_max = int((y_max_n / img_size) * orig_h)
                
                cls_id = int(cls_id)
                # Extract name and color
                name = class_names.get(cls_id, f"ID:{cls_id}")
                color = get_class_color(cls_id * 10) # Your existing function
                
                label = f"{name} {score:.2f}"
                
                # 1. Draw Bounding Box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                # 2. Draw text background
                font = cv2.FONT_HERSHEY_SIMPLEX
                (label_w, label_h), _ = cv2.getTextSize(label, font, 0.6, 2)
                cv2.rectangle(frame, (x_min, y_min - label_h - 10), (x_min + label_w + 5, y_min), color, -1)
                
                # 3. Draw Text
                cv2.putText(frame, label, (x_min + 2, y_min - 7), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅ Video saved successfully to: {output_path}")
    
    


def plot_yolo_curves(csv_path: Path, output_dir: Path, epochs_tl: int):
    """
    Generates a 2x2 dashboard of training metrics from YOLO consolidated CSV.
    
    Args:
        csv_path (Path): Path to the CSV file containing training history.
        output_dir (Path): Directory to save the resulting plot.
        epochs_tl (int): Number of epochs in the transfer learning phase (to mark transition).
    Returns:
        None
    """
    if not csv_path.exists(): return
    
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns] # Ensure clean headers
    epochs = df['epoch']
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    def add_tl_line(ax): ax.axvline(x=epochs_tl, color='gray', linestyle='--', alpha=0.7)

    df['train/total_loss'] = df['train/box_loss'] + df['train/cls_loss']
    df['val/total_loss'] = df['val/box_loss'] + df['val/cls_loss']

    # Plot 1: Losses
    axs[0, 0].plot(epochs, df['train/total_loss'], label='Train Total', color='black', linewidth=2)
    axs[0, 0].plot(epochs, df['train/box_loss'], label='Train Box', color='black', linestyle='-.')
    axs[0, 0].plot(epochs, df['train/cls_loss'], label='Train Class', color='black', linestyle='-.')
    axs[0, 0].plot(epochs, df['val/total_loss'], label='Val Total', color='red', linewidth=2)
    axs[0, 0].plot(epochs, df['val/box_loss'], label='Val Box', color='red', linestyle='-.')
    axs[0, 0].plot(epochs, df['val/cls_loss'], label='Val Class', color='red', linestyle='-.')
    add_tl_line(axs[0, 0])
    axs[0, 0].set_title('Loss Breakdown'); axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.2)

    # Plot 2: mAP
    axs[0, 1].plot(epochs, df['metrics/mAP50(B)'], label='mAP50', color='green', marker='o', markersize=3)
    axs[0, 1].plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', color='darkgreen')
    add_tl_line(axs[0, 1])
    axs[0, 1].set_title('Detection Performance'); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.2)

    # Plot 3: Learning Rate
    axs[1, 0].plot(epochs, df['lr/pg0'], color='orange', label='LR pg0')
    add_tl_line(axs[1, 0])
    axs[1, 0].set_yscale('log'); axs[1, 0].set_title('Learning Rate'); axs[1, 0].grid(True, alpha=0.2)

    # Plot 4: Metrics (Precision/Recall)
    axs[1, 1].plot(epochs, df['metrics/precision(B)'], label='Precision', color='blue')
    axs[1, 1].plot(epochs, df['metrics/recall(B)'], label='Recall', color='cyan')
    add_tl_line(axs[1, 1])
    axs[1, 1].set_title('Precision vs Recall'); axs[1, 1].legend(); axs[1, 1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_training_curves.png')
    plt.close()