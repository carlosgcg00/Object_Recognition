# src/utils/video_utils.py

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

def extract_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts metadata from a video file without loading all frames into memory.

    Args:
        video_path (Union[str, Path]): Path to the video file.

    Returns:
        Dict[str, Any]: A dictionary containing video metadata:
            - 'width' (int): Frame width.
            - 'height' (int): Frame height.
            - 'frames' (int): Total number of frames.
            - 'fps' (float): Frames per second.
            
    Raises:
        ValueError: If the video file cannot be opened.
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {path_str}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    info["duration"] = info["total_frames"] / info["fps"]
    cap.release()
    return info

def load_video_frames(
    video_path: Union[str, Path], 
    step: int = 1, 
    max_frames: int = None
) -> List[np.ndarray]:
    """
    Loads frames from a video into memory with options to sample and limit.
    
    WARNING: Use with caution on large videos to avoid Out-Of-Memory (OOM) errors.

    Args:
        video_path (Union[str, Path]): Path to the video file.
        step (int): Extracts every N-th frame. Defaults to 1 (all frames).
        max_frames (int, optional): Maximum number of frames to extract. 
                                    Defaults to None (no limit).

    Returns:
        List[np.ndarray]: A list of image frames (NumPy arrays).
        
    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {path_str}")

    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Guardamos el frame si coincide con el "step"
        if frame_idx % step == 0:
            frames.append(frame)
            
            # Detenemos el bucle si alcanzamos el límite máximo de frames
            if max_frames is not None and len(frames) >= max_frames:
                break
                
        frame_idx += 1
        
    cap.release()
    return frames