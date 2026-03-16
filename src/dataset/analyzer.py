# src/dataset/analyzer.py
from typing import Any, Dict, List


def parse_annotations(raw_lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parses MOT format ground truth lines into a list of dictionaries.
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

    Args:
        raw_lines (List[str]): Raw lines from the gt.txt file.

    Returns:
        List[Dict[str, Any]]: List of annotations with integer and float values.
    """
    annotations = []
    for line in raw_lines:
        parts = line.strip().split(',')
        if len(parts) < 7:
            continue

        ann = {
            "frame_id": int(parts[0]),
            "obj_id": int(parts[1]),
            "top_left_x": int(parts[2]),
            "top_left_y": int(parts[3]),
            "width": int(parts[4]),
            "height": int(parts[5]),
            "conf": float(parts[6]),
            "class_id": int(parts[7]) if len(parts) > 7 else -1
        }
        annotations.append(ann)
        
    return annotations