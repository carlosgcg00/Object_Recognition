# src/dataset/scanner.py

from pathlib import Path
from typing import Union, Dict, Any, List
from utils.file_utils import read_yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent

def scan_raw_dataset(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Scans the raw dataset directory based on the provided configuration file.

    Args:
        config_path (Union[str, Path]): Path to the paths.yaml configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the dataset structure:
            - 'raw_path' (Path): Root path of the raw dataset.
            - 'processed_path' (Path): Root path for the processed dataset.
            - 'classes' (Dict[str, int]): Mapping of class names to their IDs.
            - 'data' (Dict[str, List[Dict[str, Path]]]): Grouped paths per class.
              Each list contains dictionaries with 'video' and 'label' Path objects.

    Raises:
        FileNotFoundError: If the raw dataset directory does not exist.
    """
    config = read_yaml(config_path)
    
    # Extract settings from the YAML configuration
    dataset_cfg = config.get("dataset", {})
    raw_path = PROJECT_ROOT / dataset_cfg.get("raw_path", "dataset/raw")
    processed_path = PROJECT_ROOT / dataset_cfg.get("processed_path", "dataset/processed")
    classes = dataset_cfg.get("classes", {})
    structure = dataset_cfg.get("raw_structure", {})
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_path}")

    v_dir = structure.get("video_dir", "videos")
    v_ext = structure.get("video_ext", ".mp4")
    g_dir = structure.get("gt_dir", "gt")
    g_ext = structure.get("gt_ext", ".txt")

    # Initialize the result dictionary
    scanned_data = {
        "raw_path": raw_path,
        "processed_path": processed_path,
        "classes": classes,
        "data": {}
    }

    # Iterate over each class defined in the YAML
    for class_name in classes.keys():
        class_video_dir = raw_path / class_name / v_dir
        class_gt_dir = raw_path / class_name / g_dir

        # Skip if directories don't exist
        if not class_video_dir.exists() or not class_gt_dir.exists():
            print(f"Warning: Missing directories for class '{class_name}'. Skipping.")
            continue

        # Find all video files for the current class
        video_files = sorted(class_video_dir.glob(f"*{v_ext}"))
        
        class_pairs = []
        for video_path in video_files:
            # Reconstruct the expected ground truth filename (e.g., horse_1_gt.txt)
            gt_filename = f"{video_path.stem}_gt{g_ext}"
            gt_path = class_gt_dir / gt_filename

            if gt_path.exists():
                class_pairs.append({
                    "video": video_path,
                    "label": gt_path
                })
            else:
                print(f"Warning: Ground truth not found for video {video_path.name}")

        scanned_data["data"][class_name] = class_pairs

    return scanned_data

if __name__ == "__main__":
    # Quick test execution (adjust the path depending on where you run it)
    root_path = Path(__file__).parent.parent.parent
    yaml_path = root_path / "paths.yaml"
    
    if yaml_path.exists():
        dataset_info = scan_raw_dataset(yaml_path)
        print(f"Classes found: {list(dataset_info['classes'].keys())}")
        for cls_name, pairs in dataset_info['data'].items():
            print(f"  - {cls_name}: {len(pairs)} video-label pairs ready.")
    else:
        print("Ensure you are running this from a directory where paths.yaml is accessible.")