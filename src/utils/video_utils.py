import cv2
from pathlib import Path
from typing import Any, Dict, Union


def extract_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts metadata from a video file using OpenCV.

    Args:
        video_path (Union[str, Path]): Path to the .mp4 video.

    Returns:
        Dict[str, Any]: Dictionary containing width, height, fps, and frame_count.
    
    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    v_path = str(video_path)
    cap = cv2.VideoCapture(v_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {v_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    cap.release()
    return info