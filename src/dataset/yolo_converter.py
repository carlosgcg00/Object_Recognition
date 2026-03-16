# src/dataset/yolo_converter.py

import cv2
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Union

# Import our utilities and analyzer
from utils.file_utils import load_gt, save_txt
from utils.video_utils import extract_video_info
from dataset.analyzer import parse_annotations

def normalize_bbox(
    x_tl: int, 
    y_tl: int, 
    w_box: int, 
    h_box: int, 
    img_width: int, 
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Converts top-left bounding box coordinates to YOLO normalized format.

    Args:
        x_tl (int): Top-left X coordinate.
        y_tl (int): Top-left Y coordinate.
        w_box (int): Box width.
        h_box (int): Box height.
        img_width (int): Full image width.
        img_height (int): Full image height.

    Returns:
        Tuple[float, float, float, float]: Normalized (x_center, y_center, width, height).
    """
    x_center = x_tl + (w_box / 2.0)
    y_center = y_tl + (h_box / 2.0)

    # Normalize values to be between 0.0 and 1.0
    x_norm = x_center / img_width
    y_norm = y_center / img_height
    w_norm = w_box / img_width
    h_norm = h_box / img_height

    return x_norm, y_norm, w_norm, h_norm

def process_video_to_yolo(
    video_path: Union[str, Path], 
    gt_path: Union[str, Path], 
    output_img_dir: Union[str, Path], 
    output_lbl_dir: Union[str, Path],
    class_map: Dict[int, int], # <--- NUEVO ARGUMENTO AÑADIDO
    frame_step: int = 1
) -> None:
    """
    Extracts frames from a video and converts their annotations to YOLO format.

    Args:
        video_path: Path to the raw .mp4 video.
        gt_path: Path to the raw .txt ground truth.
        output_img_dir: Directory to save the extracted .jpg frames.
        output_lbl_dir: Directory to save the YOLO .txt labels.
        class_map: Dictionary to map original IDs to YOLO IDs (e.g., {6: 0, 7: 1, 8: 2}).
        frame_step: Extract every N-th frame. Defaults to 1 (all frames).
    """
    video_path = Path(video_path)
    gt_path = Path(gt_path)
    output_img_dir = Path(output_img_dir)
    output_lbl_dir = Path(output_lbl_dir)

    # Ensure output directories exist
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract video info for normalization
    v_info = extract_video_info(video_path)
    img_w, img_h = v_info['width'], v_info['height']
    base_name = video_path.stem

    # 2. Load and parse Ground Truth
    raw_gt = load_gt(gt_path)
    parsed_anns = parse_annotations(raw_gt)

    # Group annotations by frame_id
    anns_by_frame = defaultdict(list)
    for ann in parsed_anns:
        anns_by_frame[ann['frame_id']].append(ann)

    # 3. Process video frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path.name}")
        return

    frame_id = 1
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every `frame_step` frames
        if frame_id % frame_step == 0:
            # Prepare YOLO annotations for this specific frame
            yolo_lines = []
            if frame_id in anns_by_frame:
                for ann in anns_by_frame[frame_id]:
                    # ----> NUEVO: REMAPEO DEL ID AL VUELO <----
                    original_id = ann['class_id']
                    
                    # Si el ID original no está en nuestro mapa, lo saltamos para evitar errores
                    if original_id not in class_map:
                        continue
                        
                    yolo_id = class_map[original_id]

                    x_n, y_n, w_n, h_n = normalize_bbox(
                        ann['top_left_x'], ann['top_left_y'], 
                        ann['width'], ann['height'], 
                        img_w, img_h
                    )
                    
                    # Usamos el yolo_id (0, 1, 2) en lugar de ann['class_id'] (6, 7, 8)
                    yolo_lines.append(f"{yolo_id} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")

            # Define output filenames
            img_filename = f"{base_name}_frame_{frame_id}.jpg"
            lbl_filename = f"{base_name}_frame_{frame_id}.txt"
            
            img_out_path = output_img_dir / img_filename
            lbl_out_path = output_lbl_dir / lbl_filename

            # Save the image
            cv2.imwrite(str(img_out_path), frame)
            
            # Save the annotations
            save_txt(yolo_lines, lbl_out_path)
            
            extracted_count += 1

        frame_id += 1

    cap.release()
    print(f"Successfully processed {video_path.name}: {extracted_count} frames saved.")