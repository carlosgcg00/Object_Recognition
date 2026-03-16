# src/utils/utils.py
import cv2
import yaml



def read_yaml(file_path: str) -> dict:
    """Read a YAML file.
    Args:
        file_path (str): Path to the YAML file.
    Returns:
        dict: Dictionary containing the YAML data.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_video(video_path):
    """Load video.
    Args:
        video_path (str): Path to the video
    Returns:
        frames: array of images of the video
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_gt(path):
    """
    Load ground truth from a text file.

    [frame_id, targer_id, top_left_x, top_left_y, width, height, confidence, class, visibility]
    Args:
        path: path to the gt
    Returns:
        frames: array of labels of the video
    """
    with open(path, "r") as f:
        lines = f.readlines()
    
    frames = [list(map(int, ann.strip().split(','))) for ann in lines]
    return frames


def format_gt(gt):
    """
    Format ground truth from a text file.

    [frame_id, targer_id, top_left_x, top_left_y, width, height, confidence, class, visibility]
    """
    formatted_gt = []
    for ann in gt:
        formatted_gt.append({
            'frame_id': ann[0],
            'target_id': ann[1],
            'top_left_x': ann[2],
            'top_left_y': ann[3],
            'width': ann[4],
            'height': ann[5],
            'confidence': ann[6],
            'class_id': ann[7],
            'visibility': ann[8]
        })

    return formatted_gt