# src/dataset/splitter.py
import random
from pathlib import Path
from typing import Dict, List, Tuple
from utils.file_utils import save_txt
import pandas as pd


def get_processed_videos_dict(processed_dir: Path, classes: List[str]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Scans the processed/images directory and groups frames by class and video name.

    Args:
        processed_dir (Path): Path to the processed dataset directory (e.g., dataset/processed).
        classes (List[str]): List of class names to look for (e.g., ['horse', 'penguin', 'pig']).

    Returns:
        Dict[str, Dict[str, List[Path]]]: Nested dictionary structured as:
            {
                "horse": {
                    "horse_1": [Path(.../horse_1_frame_1.jpg), Path(...)],
                    "horse_2": [...]
                },
                ...
            }
    """
    img_dir = processed_dir / "images"
    dataset_dict = {cls: {} for cls in classes}
    
    # Iterate over all subdirectories
    for video_dir in img_dir.iterdir():
        if video_dir.is_dir():
            video_name = video_dir.name
            
            # Determine which class this video belongs to
            matched_class = next((cls for cls in classes if video_name.startswith(cls)), None)
            
            if matched_class:
                # Retrieve all .jpg files and sort them chronologically by frame number
                frames = list(video_dir.glob("*.jpg"))
                frames.sort(key=lambda x: int(x.stem.split('_')[-1]))
                dataset_dict[matched_class][video_name] = frames
                
    return dataset_dict

def filter_processed_data(
    data_dict: Dict[str, Dict[str, List[Path]]], 
    anomalies: List[str]
) -> Tuple[Dict[str, Dict[str, List[Path]]], Dict[str, List[str]]]:
    """
    Filters out anomalous videos from the dataset.

    Args:
        data_dict (Dict[str, Dict[str, List[Path]]]): The original dataset dictionary.
        anomalies (List[str]): List of video names (stems) to exclude.

    Returns:
        Tuple[Dict, Dict]: 
            - First element: The filtered dataset dictionary.
            - Second element: A dictionary tracking the removed anomalies per class.
    """
    filtered_data = {}
    anomalies_dict = {}
    
    for class_name, videos_dict in data_dict.items():
        # Keep videos that are NOT in the anomalies list
        valid_videos = {
            name: frames for name, frames in videos_dict.items() 
            if name not in anomalies
        }
        
        filtered_data[class_name] = valid_videos
        
        # Track which anomalies were removed for reporting purposes
        anomalies_dict[class_name] = [name for name in videos_dict.keys() if name in anomalies]
        
    return filtered_data, anomalies_dict

def get_min_frames_count(data_dict: Dict[str, Dict[str, List[Path]]]) -> int:
    """
    Finds the video with the minimum number of frames across the entire dataset.
    
    Args:
        data_dict (Dict[str, Dict[str, List[Path]]]): The dataset dictionary.

    Returns:
        int: The lowest frame count found in any single video.
    """
    min_frames = float('inf')
    for class_name, videos_dict in data_dict.items():
        for vid_name, frames in videos_dict.items():
            if len(frames) < min_frames:
                min_frames = len(frames)
    return int(min_frames)

def frame_splitter(frames: List[Path], target_frames: int = 300, step: int = None) -> List[Path]:
    """
    Sorts frames chronologically and selects the first 'target_frames' by jumping 'step' indices.
    
    Args:
        frames (List[Path]): List of frame paths.
        target_frames (int): Maximum number of frames to select.
        step (int, optional): Step size for frame selection. If None, it's calculated automatically 
                              to span the entire video. Defaults to None.
    
    Returns:
        List[Path]: A subsampled list of frame paths.
    """
    # Sort mathematically by frame ID
    frames_sorted = sorted(frames, key=lambda x: int(x.stem.split('_')[-1]))
    total = len(frames_sorted)
    
    # Return all frames if they don't exceed the target and no step is forced
    if total <= target_frames and (step is None or step <= 1):
        return frames_sorted
        
    # Calculate optimal step if not provided
    if step is None:
        step = max(1, total // target_frames)
        
    # Subsample taking 1 every 'step' frames, capped at 'target_frames'
    subsampled = frames_sorted[::step][:target_frames]
    return subsampled

def generate_balanced_kfold_yolo(
    data_dict: Dict[str, Dict[str, List[Path]]],
    output_path: Path,
    set_name: str,
    train_videos_per_class: int,
    k_folds: int = 4,
    subsample: bool = True,
    target_frames: int = 300,
    class_names: str = None,
    step: int = None
) -> None:
    """
    Generates YOLO .txt configuration files for a balanced K-Fold cross-validation setup.
    The Test set remains fixed (the last video alphabetically of each class).
    
    Args:
        data_dict (Dict): Dictionary with the processed dataset structure.
        output_path (Path): Base directory to save the experiment configuration files.
        set_name (str): Name of the experiment folder (e.g., 'set1_balanced_subsampled').
        train_videos_per_class (int): Exact number of videos to allocate for Training per fold.
        k_folds (int, optional): Number of folds to generate. Defaults to 4.
        subsample (bool, optional): If True, applies 'frame_splitter' to balance frame counts. Defaults to True.
        target_frames (int, optional): Target number of frames per video if subsampling. Defaults to 300.
        step (int, optional): Forced step jump if subsampling. Defaults to None.
    """
    out_dir = output_path / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_frames = []
    cv_data = {cls: [] for cls in data_dict.keys()}
    
    # 1. SELECT FIXED TEST SET (The last video alphabetically)
    for class_name, videos_dict in data_dict.items():
        # Sort alphabetically to ensure absolute consistency across executions
        video_names = sorted(list(videos_dict.keys()))
        
        # Select the last video for Test
        test_vid_name = video_names[-1]
        
        # Process Test frames
        frames = videos_dict[test_vid_name]
        if subsample:
            frames = frame_splitter(frames, target_frames, step)
        test_frames.extend(frames)
        
        # Keep the remaining videos for Cross-Validation (CV)
        cv_vids = video_names[:-1]
        
        if len(cv_vids) < train_videos_per_class:
            raise ValueError(
                f"Class '{class_name}' lacks sufficient videos for CV. "
                f"Available: {len(cv_vids)}, Required for Train: {train_videos_per_class}."
            )
            
        cv_data[class_name] = cv_vids
        
    # Write the fixed test set file
    test_lines = [str(p.resolve()) for p in test_frames]
    save_txt(test_lines, out_dir / "test.txt")
    
    # 2. GENERATE K-FOLDS (List Rotation Strategy)
    for i in range(k_folds):
        train_frames = []
        val_frames = []
        
        for class_name, cv_vids in cv_data.items():
            # Shift the list to rotate Train/Validation sets across folds
            shift = i % len(cv_vids)
            rotated_vids = cv_vids[shift:] + cv_vids[:shift]
            
            # Allocate Train and Validation videos
            train_vids = rotated_vids[:train_videos_per_class]
            val_vids = rotated_vids[train_videos_per_class:]
            
            # Append Train frames
            for vid in train_vids:
                frames = data_dict[class_name][vid]
                if subsample:
                    frames = frame_splitter(frames, target_frames, step)
                train_frames.extend(frames)
                
            # Append Validation frames
            for vid in val_vids:
                frames = data_dict[class_name][vid]
                if subsample:
                    frames = frame_splitter(frames, target_frames, step)
                val_frames.extend(frames)
                
        # Write Fold configurations
        train_lines = [str(p.resolve()) for p in train_frames]
        val_lines = [str(p.resolve()) for p in val_frames]
        create_yolo_yaml(
            output_path=out_dir,
            yaml_name=f"fold_{i+1}_config.yaml",
            train_txt=out_dir / f"fold_{i+1}_train.txt",
            val_txt=out_dir / f"fold_{i+1}_val.txt",
            test_txt=out_dir / "test.txt",
            class_names=class_names
        )        
        save_txt(train_lines, out_dir / f"fold_{i+1}_train.txt")
        save_txt(val_lines, out_dir / f"fold_{i+1}_val.txt")
        
    print(f"✅ Balanced YOLO configuration files generated successfully in: {out_dir}")

def generate_random_kfold_yolo(
    data_dict: Dict[str, Dict[str, List[Path]]],
    output_path: Path,
    set_name: str,
    k_folds: int = 4,
    subsample: bool = False,
    target_frames: int = 300,
    class_names: str = None,
    step: int = None
) -> None:
    """
    Generates YOLO .txt files for an unbalanced/random K-Fold setup.
    The Test set remains fixed (the last video of each class) for fair comparison 
    against balanced datasets. The remaining videos are grouped globally, shuffled, 
    and split into K folds, completely ignoring class boundaries.

    Args:
        data_dict (Dict): Dictionary with the processed dataset structure.
        output_path (Path): Base directory to save the experiment configuration files.
        set_name (str): Name of the experiment folder (e.g., 'set3_random_full').
        k_folds (int, optional): Number of folds to generate. Defaults to 4.
        subsample (bool, optional): If True, applies 'frame_splitter' to the frames. Defaults to False.
        target_frames (int, optional): Target number of frames per video if subsampling. Defaults to 300.
        step (int, optional): Forced step jump if subsampling. Defaults to None.
    """
    out_dir = output_path / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_frames = []
    cv_videos_flat = [] 
    
    # 1. SELECT FIXED TEST SET (Same logic as balanced setup)
    for class_name, videos_dict in data_dict.items():
        video_names = sorted(list(videos_dict.keys()))
        test_vid_name = video_names[-1]
        
        frames = videos_dict[test_vid_name]
        if subsample:
            frames = frame_splitter(frames, target_frames, step)
        test_frames.extend(frames)
        
        # Flatten all remaining videos into a single list of tuples (class_name, video_name)
        for v in video_names[:-1]:
            cv_videos_flat.append((class_name, v))
                
    # Write the fixed test set file
    test_lines = [str(p.resolve()) for p in test_frames]
    save_txt(test_lines, out_dir / "test.txt")
    
    # 2. GLOBAL RANDOM SHUFFLE
    rng = random.Random(42) # Set seed to ensure folds are reproducible
    rng.shuffle(cv_videos_flat)
    
    # 3. GENERATE UNBALANCED K-FOLDS
    total_cv = len(cv_videos_flat)
    fold_size = total_cv // k_folds
    
    for i in range(k_folds):
        train_frames = []
        val_frames = []
        
        # Calculate Validation split boundaries for this fold
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k_folds - 1 else total_cv
        
        val_vids = cv_videos_flat[val_start:val_end]
        train_vids = cv_videos_flat[:val_start] + cv_videos_flat[val_end:]
        
        # Extract frames for Training
        for cls_name, vid_name in train_vids:
            frames = data_dict[cls_name][vid_name]
            if subsample:
                frames = frame_splitter(frames, target_frames, step)
            train_frames.extend(frames)
            
        # Extract frames for Validation
        for cls_name, vid_name in val_vids:
            frames = data_dict[cls_name][vid_name]
            if subsample:
                frames = frame_splitter(frames, target_frames, step)
            val_frames.extend(frames)
            
        # Write Fold configurations
        train_lines = [str(p.resolve()) for p in train_frames]
        val_lines = [str(p.resolve()) for p in val_frames]
        create_yolo_yaml(
            output_path=out_dir,
            yaml_name=f"fold_{i+1}_config.yaml",
            train_txt=out_dir / f"fold_{i+1}_train.txt",
            val_txt=out_dir / f"fold_{i+1}_val.txt",
            test_txt=out_dir / "test.txt",
            class_names=class_names # ej: ['horse', 'penguin', 'pig']
        )
        save_txt(train_lines, out_dir / f"fold_{i+1}_train.txt")
        save_txt(val_lines, out_dir / f"fold_{i+1}_val.txt")
        
    print(f"✅ Random/Unbalanced YOLO configuration files generated successfully in: {out_dir}")


def get_experiment_splits_df(
    data_dict, 
    k_folds, 
    train_videos_per_class, 
    is_random=False
    ):
    
    """
    Generate a DataFrame with the assignment of videos to folds.
    
    Args:
        data_dict (Dict): Dictionary with the dataset.
        k_folds (int): Number of folds.
        train_videos_per_class (int): Number of videos per class for training.
        is_random (bool): If True, generates random splits.
    
    Returns:
        pd.DataFrame: DataFrame with the assignment of videos to folds.
    """

    all_videos = []
    test_videos = []
    
    for class_name, videos_dict in data_dict.items():
        vids = sorted(list(videos_dict.keys()))
        test_videos.append(vids[-1])
        all_videos.extend(vids)
    
    # Create base DF with all videos
    df = pd.DataFrame({'video': all_videos})
    
    # Replicate Fold logic
    for i in range(k_folds):
        fold_col = f'Fold_{i+1}'
        df[fold_col] = 'Val' # Default
        
        # Mark Test (es igual en todos los folds)
        df.loc[df['video'].isin(test_videos), fold_col] = 'Test'
        
        if not is_random:
            # Balanced logic (Rotation by class)
            for class_name, videos_dict in data_dict.items():
                cv_vids = sorted(list(videos_dict.keys()))[:-1]
                shift = i % len(cv_vids)
                rotated = cv_vids[shift:] + cv_vids[:shift]
                train_vids = rotated[:train_videos_per_class]
                
                df.loc[df['video'].isin(train_vids), fold_col] = 'Train'
        else:
            # Lógica Random (Global)
            cv_videos_flat = []
            for class_name, videos_dict in data_dict.items():
                cv_videos_flat.extend([(class_name, v) for v in sorted(list(videos_dict.keys()))[:-1]])
            
            import random
            rng = random.Random(42)
            rng.shuffle(cv_videos_flat)
            
            total_cv = len(cv_videos_flat)
            fold_size = total_cv // k_folds
            val_start = i * fold_size
            val_end = val_start + fold_size if i < k_folds - 1 else total_cv
            
            val_vids = [v[1] for v in cv_videos_flat[val_start:val_end]]
            # Los que no son Test ni Val, son Train
            df.loc[(~df['video'].isin(test_videos)) & (~df['video'].isin(val_vids)), fold_col] = 'Train'
            df.loc[df['video'].isin(val_vids), fold_col] = 'Val'

    return df.sort_values('video').reset_index(drop=True)
    

def create_yolo_yaml(
    output_path: Path, 
    yaml_name: str, 
    train_txt: Path, 
    val_txt: Path, 
    test_txt: Path, 
    class_names: List[str]
) -> None:
    """
    Generates a YOLO configuration .yaml file for a specific fold.
    
    Args:
        output_path (Path): Directory where the .yaml will be saved.
        yaml_name (str): Name of the file (e.g., 'fold_1_config.yaml').
        train_txt (Path): Absolute path to the train .txt file.
        val_txt (Path): Absolute path to the val .txt file.
        test_txt (Path): Absolute path to the test .txt file.
        class_names (List[str]): List of class names in the correct order (IDs 0, 1, 2...).
    """
    yaml_file = output_path / yaml_name
    
    # Formato estricto de YOLO
    content = [
        f"train: {train_txt.resolve()}",
        f"val: {val_txt.resolve()}",
        f"test: {test_txt.resolve()}",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}"
    ]
    
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))