# src/dataset/yolo_converter.py
import cv2
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
    """
    x_center = x_tl + (w_box / 2.0)
    y_center = y_tl + (h_box / 2.0)

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
    class_map: Dict[int, int],
    frame_step: int = 1
) -> None:
    """
    Extracts frames and converts annotations to YOLO format with ID remapping.
    """
    video_path, gt_path = Path(video_path), Path(gt_path)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    v_info = extract_video_info(video_path)
    img_w, img_h = v_info['width'], v_info['height']
    
    raw_gt = load_gt(gt_path)
    parsed_anns = parse_annotations(raw_gt)

    anns_by_frame = defaultdict(list)
    for ann in parsed_anns:
        anns_by_frame[ann['frame_id']].append(ann)

    cap = cv2.VideoCapture(str(video_path))
    frame_id = 1
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_step == 0:
            yolo_lines = []
            if frame_id in anns_by_frame:
                for ann in anns_by_frame[frame_id]:
                    original_id = ann['class_id']
                    if original_id not in class_map:
                        continue
                        
                    yolo_id = class_map[original_id]
                    x_n, y_n, w_n, h_n = normalize_bbox(
                        ann['top_left_x'], ann['top_left_y'], 
                        ann['width'], ann['height'], 
                        img_w, img_h
                    )
                    yolo_lines.append(f"{yolo_id} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")

            base_name = video_path.stem
            img_out_path = output_img_dir / f"{base_name}_frame_{frame_id}.jpg"
            lbl_out_path = output_lbl_dir / f"{base_name}_frame_{frame_id}.txt"

            cv2.imwrite(str(img_out_path), frame)
            save_txt(yolo_lines, lbl_out_path)
            extracted_count += 1

        frame_id += 1

    cap.release()
    print(f"Finished {video_path.name}: {extracted_count} frames.")


def convert_yolo_split_to_coco(
    yolo_split_txt: Path, 
    output_json: Path, 
    class_names: List[str]
) -> None:
    """
    Converts a YOLO split .txt file into a COCO format .json file.
    """
    coco_data = {
        "info": {"description": "Dataset exported from YOLO to COCO"},
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
        "images": [],
        "annotations": []
    }
    
    with open(yolo_split_txt, 'r', encoding='utf-8') as f:
        image_paths = [Path(line.strip()) for line in f.readlines() if line.strip()]
        
    ann_id = 1
    for img_id, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None: continue
            
        h, w, _ = img.shape
        coco_data["images"].append({
            "id": img_id, "file_name": str(img_path.resolve()), "width": w, "height": h
        })
        
        lbl_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt'))
        if lbl_path.exists():
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    c_id, x_c, y_c, w_n, h_n = int(parts[0]), *map(float, parts[1:5])
                    w_px, h_px = w_n * w, h_n * h
                    x_min, y_min = (x_c * w) - (w_px / 2.0), (y_c * h) - (h_px / 2.0)
                    
                    coco_data["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": c_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)],
                        "area": round(w_px * h_px, 2), "iscrowd": 0
                    })
                    ann_id += 1
                    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)