# notebook_scripts/09_tracking.py

import sys
import json
import yaml
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO

# ==========================================
# PATH CONFIGURATION
# ==========================================
# This script is intended to be executed from notebook_scripts/
PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Project imports
from utils.file_utils import read_yaml
from utils.seeds import seed_everything

from dataset.tracking_dataset import YOLOTrackingDataset

from evaluations.eval_tracking import (
    evaluate_tracker_with_label_detections,
    evaluate_label_detections_on_dataset,
    evaluate_model_detections_on_dataset,
    render_mot_summary,
)

from visualization.visualize_tracking import (
    plot_tracking_frame,
    save_tracking_sequence_frames,
    export_tracking_sequence_video,
    mot_dataframe_to_frame_dict,
)


# ==========================================
# UTILITIES
# ==========================================
SEED = 42
seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# CONFIGURATION FILE
# ==========================================
CONFIG_PATH = PROJECT_ROOT / "paths.yaml"
config = read_yaml(CONFIG_PATH)

PROCESSED_DIR = PROJECT_ROOT / config["dataset"]["processed_path"]

class_order = list(config["dataset"]["classes"].keys())
CLASS_NAMES = {idx: name for idx, name in enumerate(class_order)}


# ==========================================
# SELECT YOLO DETECTOR MODEL
# ==========================================
SELECTED_MODEL = "yolo26m"
SELECTED_EXPERIMENT = "set2_balanced_full"
N_FOLD = 2
SELECTED_FOLD = f"fold_{N_FOLD}_config"

MODEL_RUN_DIR = PROJECT_ROOT / "runs" / SELECTED_MODEL / SELECTED_EXPERIMENT / SELECTED_FOLD
MODEL_PATH = MODEL_RUN_DIR / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# ==========================================
# DATASET CONFIGURATION
# ==========================================
IMAGE_ROOT_NAME = "images"
LABEL_ROOT_NAME = "labels_track"

SPLIT_FILES = [
    f"fold_{N_FOLD}_train.txt",
    f"fold_{N_FOLD}_val.txt",
    "test.txt",
]


# ==========================================
# MOTMETRICS CONFIGURATION
# ==========================================
TRACKING_METRICS = [
    "num_frames",
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
]

IOU_MATCH_THRESHOLD = 0.5

# ==========================================
# TRACKER CONFIGURATION
# ==========================================
REID_MODEL_PATH = "yolo26n-cls.pt"
tracker_args = {
    "tracker_type": "bytetrack",  # "bytetrack" or "botsort"

    # Common ByteTrack / BoT-SORT parameters
    "match_thresh": 0.4,
    "new_track_thresh": 0.5,
    "track_buffer": 50,
    "track_high_thresh": 0.8,
    "track_low_thresh": 0.2,
    "fuse_score": True,

    # BoT-SORT parameters required by Ultralytics
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": True,
    "model": str(REID_MODEL_PATH),
}

TRACKER_TYPE = tracker_args["tracker_type"]

# ==========================================
# TRACKING OUTPUT DIRECTORY
# ==========================================
RUN_NAME = f"{tracker_args['tracker_type']}_tracking"

TRACKING_ROOT = MODEL_RUN_DIR / "tracking" / RUN_NAME
TRACKING_ROOT.mkdir(parents=True, exist_ok=True)


# ==========================================
# DETECTOR CONFIGURATION
# ==========================================
DET_CONF = 0.25


# ==========================================
# VIDEO EXPORT CONFIGURATION
# ==========================================
FPS_EXPORT = 10.0


# ==========================================
# UTILS FUNCTIONS
# ==========================================
def get_split_path(split_file: str) -> Path:
    """
    Builds the absolute path to a split file.

    Args:
        split_file (str): Split filename, for example
            'fold_2_train.txt', 'fold_2_val.txt' or 'test.txt'.

    Returns:
        Path: Absolute path to the split file.
    """
    return PROCESSED_DIR / "experiments" / SELECTED_EXPERIMENT / split_file


def build_tracking_dataset(split_txt: Path) -> YOLOTrackingDataset:
    """
    Builds a YOLOTrackingDataset for a split.

    Args:
        split_txt (Path): Path to the split .txt file.

    Returns:
        YOLOTrackingDataset: Tracking dataset instance.
    """
    return YOLOTrackingDataset(
        split_txt=split_txt,
        image_root_name=IMAGE_ROOT_NAME,
        label_root_name=LABEL_ROOT_NAME,
        rgb=True,
        return_tensors=True,
        seed=SEED,
    )


def save_tracking_run_config(
    output_dir: Path,
    split_txt: Path,
    track_dataset: YOLOTrackingDataset,
) -> None:
    """
    Saves the run configuration as YAML and JSON.

    Args:
        output_dir (Path): Directory where config files will be saved.
        split_txt (Path): Split file used for the current run.
        track_dataset (YOLOTrackingDataset): Dataset instance used in this run.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "seed": SEED,
        "device": DEVICE,
        "project_root": str(PROJECT_ROOT),
        "processed_dir": str(PROCESSED_DIR),
        "split_txt": str(split_txt),
        "image_root_name": IMAGE_ROOT_NAME,
        "label_root_name": LABEL_ROOT_NAME,
        "selected_model": SELECTED_MODEL,
        "selected_experiment": SELECTED_EXPERIMENT,
        "selected_fold": SELECTED_FOLD,
        "model_run_dir": str(MODEL_RUN_DIR),
        "model_path": str(MODEL_PATH),
        "tracking_root": str(TRACKING_ROOT),
        "run_name": RUN_NAME,
        "det_conf": DET_CONF,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "tracking_metrics": TRACKING_METRICS,
        "tracker_args": tracker_args,
        "class_names": CLASS_NAMES,
        "dataset_summary": track_dataset.summary(),
    }

    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            run_config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            run_config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Saved run config to: {output_dir}")


def save_sample_gt_outputs(
    track_dataset: YOLOTrackingDataset,
    split_output_dir: Path,
) -> str:
    """
    Saves GT tracking visualizations for the first sequence in the split.

    This creates:
        - one sample PNG/PDF frame
        - all annotated GT frames for the selected sequence
        - one MP4 video with GT trajectories

    Args:
        track_dataset (YOLOTrackingDataset): Dataset for the current split.
        split_output_dir (Path): Output directory for the current split.

    Returns:
        str: Selected sequence name.
    """
    seq_name = track_dataset.sequence_names[0]

    image, target = next(track_dataset.iter_sequence(seq_name))

    sample_dir = split_output_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_img_path = sample_dir / f"{seq_name}_tracking_frame.png"

    plot_tracking_frame(
        image=image,
        target=target,
        pred_df=None,
        class_names=CLASS_NAMES,
        title=f"GT tracking | {seq_name} | frame={target['frame_number']}",
        save_path=save_img_path,
    )

    gt_vis_dir = split_output_dir / "00_gt_visualization"
    gt_frames_dir = gt_vis_dir / "frames" / seq_name
    gt_video_dir = gt_vis_dir / "videos"

    gt_frames_dir.mkdir(parents=True, exist_ok=True)
    gt_video_dir.mkdir(parents=True, exist_ok=True)

    save_tracking_sequence_frames(
        dataset=track_dataset,
        sequence_name=seq_name,
        output_dir=gt_frames_dir,
        pred_by_frame=None,
        class_names=CLASS_NAMES,
        use_compact_frame_index=True,
    )

    gt_video_path = gt_video_dir / f"{seq_name}_gt.mp4"

    export_tracking_sequence_video(
        dataset=track_dataset,
        sequence_name=seq_name,
        output_path=gt_video_path,
        pred_by_frame=None,
        class_names=CLASS_NAMES,
        fps=FPS_EXPORT,
        use_compact_frame_index=True,
    )

    print(f"Saved sample GT frame to: {save_img_path}")
    print(f"Saved GT frames to: {gt_frames_dir}")
    print(f"Saved GT video to: {gt_video_path}")

    return seq_name


def run_tracking_with_gt_label_detections(
    track_dataset: YOLOTrackingDataset,
    split_output_dir: Path,
    seq_name: str,
) -> pd.DataFrame:
    """
    Runs tracking using GT labels as detector inputs.

    This phase evaluates the tracker using labels_track as perfect detections.

    Args:
        track_dataset (YOLOTrackingDataset): Dataset for the current split.
        split_output_dir (Path): Output directory for the current split.
        seq_name (str): Sequence name used for sample video export.

    Returns:
        pd.DataFrame: Metrics summary for all sequences in the split.
    """
    phase_dir = split_output_dir / "01_tracking_gt_label_detections" / TRACKER_TYPE
    phase_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # Sample sequence with video
    # ------------------------------------------
    sample_seq_dir = phase_dir / "sample_sequence" / seq_name
    sample_seq_dir.mkdir(parents=True, exist_ok=True)

    summary_seq, gt_df_seq, pred_df_seq, _ = evaluate_tracker_with_label_detections(
        dataset=track_dataset,
        sequence_name=seq_name,
        tracker_args=tracker_args,
        metrics=TRACKING_METRICS,
        iou_match_threshold=IOU_MATCH_THRESHOLD,
        output_dir=sample_seq_dir,
    )

    print("\nGT label detections | sample sequence")
    print(render_mot_summary(summary_seq))

    pred_by_frame = mot_dataframe_to_frame_dict(pred_df_seq)

    video_dir = phase_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"{seq_name}_tracking_gt_detections.mp4"

    export_tracking_sequence_video(
        dataset=track_dataset,
        sequence_name=seq_name,
        output_path=video_path,
        pred_by_frame=pred_by_frame,
        class_names=CLASS_NAMES,
        fps=FPS_EXPORT,
        use_compact_frame_index=True,
    )

    print(f"Saved GT-label tracker video to: {video_path}")

    # ------------------------------------------
    # All sequences
    # ------------------------------------------
    all_sequences_dir = phase_dir / "all_sequences"
    all_sequences_dir.mkdir(parents=True, exist_ok=True)

    metrics_all, gt_by_seq, pred_by_seq = evaluate_label_detections_on_dataset(
        dataset=track_dataset,
        tracker_args=tracker_args,
        metrics=TRACKING_METRICS,
        iou_match_threshold=IOU_MATCH_THRESHOLD,
        output_dir=all_sequences_dir,
    )

    metrics_path = all_sequences_dir / "metrics_tracking_gt_all_sequences.csv"
    metrics_all.to_csv(metrics_path, index=True)

    print("\nGT label detections | all sequences")
    print(render_mot_summary(metrics_all))
    print(f"Saved metrics to: {metrics_path}")

    return metrics_all


def run_tracking_with_model_detections(
    track_dataset: YOLOTrackingDataset,
    split_output_dir: Path,
) -> pd.DataFrame:
    """
    Runs YOLO detections followed by tracking on all sequences.

    Args:
        track_dataset (YOLOTrackingDataset): Dataset for the current split.
        split_output_dir (Path): Output directory for the current split.

    Returns:
        pd.DataFrame: Metrics summary for all sequences in the split.
    """
    phase_dir = split_output_dir / "02_model_detections" / TRACKER_TYPE
    phase_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    all_sequences_dir = phase_dir / "all_sequences"
    all_sequences_dir.mkdir(parents=True, exist_ok=True)

    metrics_all, gt_by_seq, pred_by_seq = evaluate_model_detections_on_dataset(
        model=model,
        dataset=track_dataset,
        tracker_args=tracker_args,
        metrics=TRACKING_METRICS,
        conf=DET_CONF,
        iou_match_threshold=IOU_MATCH_THRESHOLD,
        output_dir=all_sequences_dir,
    )

    metrics_path = all_sequences_dir / "metrics_tracking_model_detections_all_sequences.csv"
    metrics_all.to_csv(metrics_path, index=True)

    print("\nModel detections | all sequences")
    print(render_mot_summary(metrics_all))
    print(f"Saved metrics to: {metrics_path}")

    return metrics_all


def run_split(split_txt: Path) -> None:
    """
    Runs the full tracking pipeline for one split.

    Args:
        split_txt (Path): Path to the split file.

    Returns:
        None
    """
    if not split_txt.exists():
        print(f"Skipping missing split: {split_txt}")
        return

    split_name = split_txt.stem
    split_output_dir = TRACKING_ROOT / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Running tracking split: {split_name}")
    print(f"Split file: {split_txt}")
    print(f"Output dir: {split_output_dir}")
    print("=" * 100)

    track_dataset = build_tracking_dataset(split_txt)

    save_tracking_run_config(
        output_dir=split_output_dir,
        split_txt=split_txt,
        track_dataset=track_dataset,
    )

    seq_name = save_sample_gt_outputs(
        track_dataset=track_dataset,
        split_output_dir=split_output_dir,
    )

    metrics_gt = run_tracking_with_gt_label_detections(
        track_dataset=track_dataset,
        split_output_dir=split_output_dir,
        seq_name=seq_name,
    )

    metrics_model = run_tracking_with_model_detections(
        track_dataset=track_dataset,
        split_output_dir=split_output_dir,
    )

    comparison = pd.concat(
        {
            "gt_label_detections": metrics_gt,
            "model_detections": metrics_model,
        },
        names=["source", "sequence"],
    )

    comparison_path = split_output_dir / "comparison_gt_vs_model_detections.csv"
    comparison.to_csv(comparison_path, index=True)

    print("\nComparison | GT label detections vs model detections")
    print(render_mot_summary(comparison))
    print(f"Saved comparison to: {comparison_path}")


def main() -> None:
    """
    Runs the tracking pipeline for all configured splits.

    Args:
        None

    Returns:
        None
    """
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEVICE:", DEVICE)
    print("MODEL_PATH:", MODEL_PATH)
    print("TRACKING_ROOT:", TRACKING_ROOT)
    print("CLASS_NAMES:", CLASS_NAMES)
    print("TRACKER_TYPE:", TRACKER_TYPE)
    print("TRACKER_ARGS:", tracker_args)

    for split_file in SPLIT_FILES:
        print(f"\n=> Configuring tracking dataset for split: {split_file}")
        split_txt = get_split_path(split_file)
        run_split(split_txt)


if __name__ == "__main__":
    main()