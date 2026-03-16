# src/dataset/splitter.py

import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

from utils.file_utils import save_txt


def get_processed_videos_dict(processed_dir: Path, classes: List[str]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Scans the processed/images directory and groups frames by class and video name.

    Args:
        processed_dir (Path): Path to the processed directory.
        classes (List[str]): List of class names.

    Returns:
        Dict[str, Dict[str, List[Path]]]: Nested dictionary with classes and videos.
    """
    img_dir = processed_dir / "images"
    dataset_dict = {cls: {} for cls in classes}

    for video_dir in img_dir.iterdir():
        if video_dir.is_dir():
            video_name = video_dir.name
            matched_class = next((cls for cls in classes if video_name.startswith(cls)), None)

            if matched_class:
                frames = list(video_dir.glob("*.jpg"))
                # Sort numerically using the frame ID at the end of the filename
                frames.sort(key=lambda x: int(x.stem.split('_')[-1]))
                dataset_dict[matched_class][video_name] = frames

    return dataset_dict


def filter_processed_data(
    data_dict: Dict[str, Dict[str, List[Path]]], 
    anomalies: List[str]
) -> Tuple[Dict[str, Dict[str, List[Path]]], Dict[str, List[str]]]:
    """
    Filters out anomalous videos from the data dictionary.

    Args:
        data_dict (Dict): The original data dictionary.
        anomalies (List[str]): Video names to exclude.

    Returns:
        Tuple[Dict, Dict]: Filtered data and a dictionary of removed anomalies.
    """
    filtered_data = {}
    anomalies_dict = {}

    for class_name, videos_dict in data_dict.items():
        valid_videos = {name: f for name, f in videos_dict.items() if name not in anomalies}
        filtered_data[class_name] = valid_videos
        anomalies_dict[class_name] = [n for n in videos_dict.keys() if n in anomalies]

    return filtered_data, anomalies_dict


def get_min_frames_count(data_dict: Dict[str, Dict[str, List[Path]]]) -> int:
    """
    Finds the minimum number of frames in any single video within the dataset.
    """
    min_frames = float('inf')
    for class_name, videos_dict in data_dict.items():
        for vid_name, frames in videos_dict.items():
            if len(frames) < min_frames:
                min_frames = len(frames)
    return int(min_frames)


def frame_splitter(frames: List[Path], target_frames: int = 300, step: int = None) -> List[Path]:
    """
    Subsamples frames to reach target_frames using a calculated or fixed step.
    """
    total = len(frames)
    if total <= target_frames and (step is None or step <= 1):
        return frames

    if step is None:
        step = max(1, total // target_frames)

    return frames[::step][:target_frames]


def create_yolo_yaml(
    output_path: Path, 
    yaml_name: str, 
    train_txt: Path, 
    val_txt: Path, 
    test_txt: Path, 
    class_names: List[str]
) -> None:
    """
    Generates a YOLOv8/v11 compatible .yaml configuration file.
    """
    content = [
        f"train: {train_txt.resolve()}",
        f"val: {val_txt.resolve()}",
        f"test: {test_txt.resolve()}",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}"
    ]
    with open(output_path / yaml_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))


def generate_balanced_kfold_yolo(
    data_dict: Dict[str, Dict[str, List[Path]]],
    output_path: Path,
    set_name: str,
    train_videos_per_class: int,
    class_names: List[str],
    k_folds: int = 4,
    subsample: bool = True,
    target_frames: int = 300
) -> None:
    """
    Creates a balanced K-Fold setup where classes and frame counts are equalized.
    The last video of each class is reserved for Test.
    """
    out_dir = output_path / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_frames = []
    cv_data = {cls: [] for cls in data_dict.keys()}

    # 1. Fixed Test Set (Last video alphabetically)
    for class_name, videos_dict in data_dict.items():
        video_names = sorted(list(videos_dict.keys()))
        test_vid = video_names[-1]
        
        frames = videos_dict[test_vid]
        if subsample:
            frames = frame_splitter(frames, target_frames)
        test_frames.extend(frames)
        
        cv_data[class_name] = video_names[:-1]

    save_txt([str(p.resolve()) for p in test_frames], out_dir / "test.txt")

    # 2. Folds generation
    for i in range(k_folds):
        train_f, val_f = [], []
        for class_name, cv_vids in cv_data.items():
            shift = i % len(cv_vids)
            rotated = cv_vids[shift:] + cv_vids[:shift]
            
            for v in rotated[:train_videos_per_class]:
                f = data_dict[class_name][v]
                train_f.extend(frame_splitter(f, target_frames) if subsample else f)
            
            for v in rotated[train_videos_per_class:]:
                f = data_dict[class_name][v]
                val_f.extend(frame_splitter(f, target_frames) if subsample else f)

        t_path, v_path = out_dir / f"fold_{i+1}_train.txt", out_dir / f"fold_{i+1}_val.txt"
        save_txt([str(p.resolve()) for p in train_f], t_path)
        save_txt([str(p.resolve()) for p in val_f], v_path)
        
        create_yolo_yaml(out_dir, f"fold_{i+1}_config.yaml", t_path, v_path, out_dir / "test.txt", class_names)


def generate_random_kfold_yolo(
    data_dict: Dict[str, Dict[str, List[Path]]],
    output_path: Path,
    set_name: str,
    class_names: List[str],
    k_folds: int = 4,
    subsample: bool = False,
    target_frames: int = 300
) -> None:
    """
    Creates an unbalanced K-Fold setup with globally shuffled videos.
    The Test set remains fixed (last video per class) for consistency.
    """
    out_dir = output_path / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_frames, cv_vids_flat = [], []
    for class_name, videos_dict in data_dict.items():
        names = sorted(list(videos_dict.keys()))
        test_vid = names[-1]
        
        f = videos_dict[test_vid]
        test_frames.extend(frame_splitter(f, target_frames) if subsample else f)
        
        for v in names[:-1]:
            cv_vids_flat.append((class_name, v))

    save_txt([str(p.resolve()) for p in test_frames], out_dir / "test.txt")

    random.Random(42).shuffle(cv_vids_flat)
    f_size = len(cv_vids_flat) // k_folds

    for i in range(k_folds):
        train_f, val_f = [], []
        v_start = i * f_size
        v_end = v_start + f_size if i < k_folds - 1 else len(cv_vids_flat)
        
        for cls, vid in cv_vids_flat[:v_start] + cv_vids_flat[v_end:]:
            f = data_dict[cls][vid]
            train_f.extend(frame_splitter(f, target_frames) if subsample else f)
            
        for cls, vid in cv_vids_flat[v_start:v_end]:
            f = data_dict[cls][vid]
            val_f.extend(frame_splitter(f, target_frames) if subsample else f)

        t_path, v_path = out_dir / f"fold_{i+1}_train.txt", out_dir / f"fold_{i+1}_val.txt"
        save_txt([str(p.resolve()) for p in train_f], t_path)
        save_txt([str(p.resolve()) for p in val_f], v_path)
        
        create_yolo_yaml(out_dir, f"fold_{i+1}_config.yaml", t_path, v_path, out_dir / "test.txt", class_names)