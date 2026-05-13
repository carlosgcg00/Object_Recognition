# src/utils/file_utils.py

import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Reads a YAML configuration file.

    Args:
        file_path (Union[str, Path]): The path to the YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed YAML data.
        
    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_gt(file_path: Union[str, Path]) -> List[List[int]]:
    """
    Loads ground truth annotations from a text file.
    
    Expected format per line:
    [frame_id, target_id, top_left_x, top_left_y, width, height, confidence, class, visibility]

    Args:
        file_path (Union[str, Path]): Path to the ground truth .txt file.

    Returns:
        List[List[int]]: A list of lists, where each inner list represents a parsed annotation line.
    """
    with open(Path(file_path), "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Strip whitespace and split by comma, then convert to integers
    annotations = [list(map(int, line.strip().split(','))) for line in lines if line.strip()]
    return annotations

def save_txt(data: List[str], file_path: Union[str, Path]) -> None:
    """
    Saves a list of strings to a text file.

    Args:
        data (List[str]): Data to be written, where each string is a line.
        file_path (Union[str, Path]): The destination path for the text file.
    """
    with open(Path(file_path), "w", encoding="utf-8") as file:
        for line in data:
            file.write(f"{line}\n")

def save_image(frame: np.ndarray, file_path: Union[str, Path]) -> None:
    """
    Saves an image array to disk.

    Args:
        frame (np.ndarray): The image data to save (usually in BGR format for OpenCV).
        file_path (Union[str, Path]): The destination path for the image.
    """
    # OpenCV requires strings for paths in some underlying C++ implementations
    cv2.imwrite(str(file_path), frame)
    

def build_ordered_class_dict(dataset_names: List[str], class_order: List[str], one_based: bool = False) -> Dict[int, str]:
    """
    Builds an ordered {class_id: class_name} dict following class_order.

    Args:
        dataset_names (List[str]): Names from dataset YAML.
        class_order (List[str]): Desired order from paths.yaml.
        one_based (bool): True for EfficientDet (1..N), False for YOLO (0..N-1).

    Returns:
        Dict[int, str]: Ordered mapping preserving insertion order.
    """
    offset = 1 if one_based else 0
    name_to_id = {name: i + offset for i, name in enumerate(dataset_names)}

    ordered = {}

    # First, classes appearing in paths.yaml order
    for name in class_order:
        if name in name_to_id:
            ordered[name_to_id[name]] = name

    # Then any remaining classes not listed in paths.yaml
    for name in dataset_names:
        if name not in class_order:
            ordered[name_to_id[name]] = name

    return ordered