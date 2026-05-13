# src/dataset/analyzer.py

import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

# Import our new utilities
from utils.file_utils import load_gt
from utils.video_utils import extract_video_info

def parse_annotations(gt_raw: List[List[int]]) -> List[Dict[str, int]]:
    """
    Converts raw ground truth lists into structured dictionaries.
    
    Expected raw format:
    [frame_id, target_id, top_left_x, top_left_y, width, height, confidence, class, visibility]

    Args:
        gt_raw (List[List[int]]): Raw annotations loaded from the text file.

    Returns:
        List[Dict[str, int]]: A list of dictionaries with descriptive keys.
    """
    parsed_gt = []
    for ann in gt_raw:
        # Ensure the row has the minimum required columns to avoid IndexError
        if len(ann) >= 9:
            parsed_gt.append({
                'frame_id': ann[0],
                'target_id': ann[1],
                'top_left_x': ann[2],
                'top_left_y': ann[3],
                'bbox_width': ann[4],
                'bbox_height': ann[5],
                'confidence': ann[6],
                'class_id': ann[7],
                'visibility': ann[8]
            })
    return parsed_gt

def generate_dataset_summary(scanned_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates a summary DataFrame containing video metadata and annotation counts per frame.

    Args:
        scanned_data (Dict[str, Any]): The dataset structure returned by scan_raw_dataset().

    Returns:
        pd.DataFrame: A formatted DataFrame containing the merged dataset information.
                      Returns an empty DataFrame if no data is found.
    """
    all_data_list = []
    
    dataset_dict = scanned_data.get("data", {})
    
    for class_name, pairs in dataset_dict.items():
        for pair in pairs:
            video_path: Path = pair['video']
            gt_path: Path = pair['label']
            
            # 1. Extract video metadata
            try:
                v_info = extract_video_info(video_path)
            except Exception as e:
                print(video_path)
                print(f"Error reading video {video_path.name}: {e}")
                continue
                
            # 2. Load and parse annotations
            try:
                raw_annotations = load_gt(gt_path)
                formatted_annotations = parse_annotations(raw_annotations)
            except Exception as e:
                print(f"Error reading GT {gt_path.name}: {e}")
                continue
            
            # 3. Process into a DataFrame
            df_temp = pd.DataFrame(formatted_annotations)
            
            if not df_temp.empty:
                # Group by frame_id to get the object count per frame
                df_counts = df_temp.groupby('frame_id').size().reset_index(name='object_count')
                
                # Inject video metadata into the DataFrame
                df_counts['video_name'] = video_path.stem
                df_counts['class_name'] = class_name
                df_counts['width'] = v_info['width']
                df_counts['height'] = v_info['height']
                df_counts['total_frames'] = v_info['total_frames']
                df_counts['fps'] = v_info['fps']
                df_counts['duration_sec'] = v_info['duration']
                
                all_data_list.append(df_counts)

    # 4. Concatenate all processed data
    if all_data_list:
        df_final = pd.concat(all_data_list, ignore_index=True)
        
        # Reorder columns for better readability in notebooks
        column_order = [
            'video_name', 'class_name', 'frame_id', 'object_count', 
            'width', 'height', 'total_frames', 'fps', 'duration_sec'
        ]
        return df_final[column_order]
    
    return pd.DataFrame()