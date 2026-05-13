# src/evaluations/eval_tracking.py

from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from types import SimpleNamespace
from time import perf_counter

import cv2
import torch
import numpy as np
import pandas as pd
import motmetrics as mm

from ultralytics import YOLO
from ultralytics.trackers import BYTETracker, BOTSORT


# Compatibility patch for motmetrics + NumPy >= 2.0
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


TRACK_COLS = [
    "frame",
    "object_id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
]


DEFAULT_MOT_METRICS = [
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


TIMING_COLS = [
    "frame",
    "tiempo_detector_sec",
    "tiempo_tracker_sec",
]


def _sync_cuda_if_available() -> None:
    """
    Synchronizes CUDA operations if CUDA is available.

    Args:
        None

    Returns:
        None
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_numpy(value: Any) -> np.ndarray:
    """
    Converts tensors or array-like objects to NumPy arrays.

    Args:
        value (Any): Tensor, NumPy array, list, or array-like object.

    Returns:
        np.ndarray: Value converted to a NumPy array.
    """
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _summarize_timing_dataframe(timing_df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarizes per-frame timing information.

    Args:
        timing_df (pd.DataFrame): Dataframe with per-frame detector and tracker
            timings.

    Returns:
        Dict[str, float]: Dictionary with:
            - tiempo_detector/frame
            - tiempo_tracker/frame
    """
    if timing_df is None or timing_df.empty:
        return {
            "tiempo_detector/frame": np.nan,
            "tiempo_tracker/frame": np.nan,
        }

    return {
        "tiempo_detector/frame": float(timing_df["tiempo_detector_sec"].mean()),
        "tiempo_tracker/frame": float(timing_df["tiempo_tracker_sec"].mean()),
    }


def _add_timing_to_summary(
    summary: pd.DataFrame,
    timing_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds timing summary columns to a motmetrics summary dataframe.

    Args:
        summary (pd.DataFrame): MOT metrics summary.
        timing_df (pd.DataFrame): Per-frame timing dataframe.

    Returns:
        pd.DataFrame: Summary dataframe with timing columns added.
    """
    timing_summary = _summarize_timing_dataframe(timing_df)

    for key, value in timing_summary.items():
        summary[key] = value

    return summary


def xyxy_to_xywh_center_abs(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Converts absolute xyxy boxes to absolute xywh-center boxes.

    Args:
        boxes_xyxy (np.ndarray): Array with shape (N, 4), using
            [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Array with shape (N, 4), using
            [x_center, y_center, width, height].
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)

    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x_min = boxes_xyxy[:, 0]
    y_min = boxes_xyxy[:, 1]
    x_max = boxes_xyxy[:, 2]
    y_max = boxes_xyxy[:, 3]

    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    return np.stack([x_center, y_center, width, height], axis=1).astype(np.float32)


def xyxy_to_mot_xywh_abs(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Converts absolute xyxy boxes to absolute MOT-style xywh boxes.

    Args:
        boxes_xyxy (np.ndarray): Array with shape (N, 4), using
            [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Array with shape (N, 4), using
            [bb_left, bb_top, bb_width, bb_height].
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)

    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x_min = boxes_xyxy[:, 0]
    y_min = boxes_xyxy[:, 1]
    width = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    height = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return np.stack([x_min, y_min, width, height], axis=1).astype(np.float32)


def build_tracker(tracker_args: Dict[str, Any]):
    """
    Builds an Ultralytics BYTETracker or BOTSORT instance from a dictionary.

    Args:
        tracker_args (Dict[str, Any]): Tracker configuration dictionary.
            Expected key:
                - tracker_type: "bytetrack" or "botsort"

            Common ByteTrack/BoT-SORT keys:
                - match_thresh
                - new_track_thresh
                - track_buffer
                - track_high_thresh
                - track_low_thresh
                - fuse_score

            BoT-SORT-specific optional keys:
                - with_reid
                - gmc_method
                - proximity_thresh
                - appearance_thresh
                - model

    Returns:
        BYTETracker | BOTSORT: Instantiated tracker.
    """
    tracker_type = str(tracker_args.get("tracker_type", "bytetrack")).lower()
    args = SimpleNamespace(**tracker_args)

    if tracker_type == "bytetrack":
        return BYTETracker(args)

    if tracker_type == "botsort":
        return BOTSORT(args)

    raise ValueError(f"Unsupported tracker_type: {tracker_type}")


class TrackerDetections:
    """
    Lightweight detection container compatible with Ultralytics BYTETracker/BOTSORT.

    Ultralytics trackers expect an object with:
        - xyxy
        - xywh
        - conf
        - cls

    and, in recent versions, they also expect the object to support indexing:
        results[mask]

    Args:
        xyxy (np.ndarray): Array with shape (N, 4), using [x1, y1, x2, y2].
        xywh (np.ndarray): Array with shape (N, 4), using [xc, yc, w, h].
        conf (np.ndarray): Array with shape (N,), detection confidences.
        cls (np.ndarray): Array with shape (N,), class IDs.

    Returns:
        TrackerDetections: Detection container compatible with tracker.update().
    """

    def __init__(
        self,
        xyxy: np.ndarray,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

        n = len(self.conf)

        if len(self.xyxy) != n:
            raise ValueError(f"xyxy length {len(self.xyxy)} != conf length {n}")

        if len(self.xywh) != n:
            raise ValueError(f"xywh length {len(self.xywh)} != conf length {n}")

        if len(self.cls) != n:
            raise ValueError(f"cls length {len(self.cls)} != conf length {n}")

    def __len__(self) -> int:
        """
        Returns the number of detections.

        Args:
            None

        Returns:
            int: Number of detections.
        """
        return len(self.conf)

    def __getitem__(self, idx):
        """
        Returns a subset of detections.

        Args:
            idx: Integer index, slice, list of indices, or boolean mask.

        Returns:
            TrackerDetections: Subset detection container.
        """
        return TrackerDetections(
            xyxy=self.xyxy[idx],
            xywh=self.xywh[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
        )


def make_gt_mot_dataframe_for_sequence(
    dataset,
    sequence_name: str,
    use_compact_frame_index: bool = True,
) -> pd.DataFrame:
    """
    Converts one YOLOTrackingDataset sequence to a MOT-like GT dataframe.

    Args:
        dataset: YOLOTrackingDataset-like object. It must implement
            get_sequence_indices(sequence_name) and __getitem__.
        sequence_name (str): Name of the sequence to convert.
        use_compact_frame_index (bool): If True, converts original frame numbers
            such as 5, 10, 15 into compact frame IDs 1, 2, 3.

    Returns:
        pd.DataFrame: Ground-truth dataframe with TRACK_COLS plus metadata.
    """
    rows = []

    sequence_indices = dataset.get_sequence_indices(sequence_name)
    frame_number_to_compact = {}

    if use_compact_frame_index:
        for compact_frame, idx in enumerate(sequence_indices, start=1):
            _, target = dataset[idx]
            original_frame_number = int(target["frame_number"])
            frame_number_to_compact[original_frame_number] = compact_frame

    for idx in sequence_indices:
        _, target = dataset[idx]

        original_frame_number = int(target["frame_number"])

        if use_compact_frame_index:
            frame = frame_number_to_compact[original_frame_number]
        else:
            frame = original_frame_number

        boxes_xywh_abs = _to_numpy(target["boxes_xywh_abs"])
        labels = _to_numpy(target["labels"])
        track_ids = _to_numpy(target["track_ids"])

        for box, class_id, track_id in zip(boxes_xywh_abs, labels, track_ids):
            bb_left, bb_top, bb_width, bb_height = map(float, box)

            if bb_width <= 0 or bb_height <= 0:
                continue

            rows.append(
                {
                    "frame": int(frame),
                    "object_id": int(track_id),
                    "bb_left": bb_left,
                    "bb_top": bb_top,
                    "bb_width": bb_width,
                    "bb_height": bb_height,
                    "conf": 1.0,
                    "x": -1.0,
                    "y": -1.0,
                    "z": -1.0,
                    "class_id": int(class_id),
                    "sequence_name": sequence_name,
                    "original_frame_number": original_frame_number,
                    "image_path": target["image_path"],
                    "label_path": target["label_path"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=TRACK_COLS
            + [
                "class_id",
                "sequence_name",
                "original_frame_number",
                "image_path",
                "label_path",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["frame", "object_id"])
        .reset_index(drop=True)
    )


def get_label_detections_for_sequence(
    dataset,
    sequence_name: str,
    use_compact_frame_index: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Creates detections from labels_track for one sequence.

    Args:
        dataset: YOLOTrackingDataset-like object.
        sequence_name (str): Sequence name.
        use_compact_frame_index (bool): If True, frame keys are 1, 2, 3...
            following the extracted sequence order.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary indexed by frame ID.
    """
    detections = {}
    sequence_indices = dataset.get_sequence_indices(sequence_name)

    for compact_frame, idx in enumerate(sequence_indices, start=1):
        image, target = dataset[idx]

        original_frame_number = int(target["frame_number"])
        frame = compact_frame if use_compact_frame_index else original_frame_number

        boxes_xyxy = _to_numpy(target["boxes_xyxy_abs"]).astype(np.float32).reshape(-1, 4)
        labels = _to_numpy(target["labels"]).astype(np.float32).reshape(-1)

        boxes_xywh_center = xyxy_to_xywh_center_abs(boxes_xyxy)
        confs = np.ones((len(boxes_xyxy),), dtype=np.float32)

        detections[int(frame)] = {
            "xyxy": boxes_xyxy,
            "xywh": boxes_xywh_center,
            "cls": labels,
            "conf": confs,
            "image": image,
            "image_path": target["image_path"],
            "original_frame_number": original_frame_number,
            "tiempo_detector_sec": 0.0,
        }

    return detections


def get_yolo_detections_for_sequence(
    model: YOLO,
    dataset,
    sequence_name: str,
    conf: float = 0.25,
    imgsz: int = 640,
    target_class_id: Optional[int] = None,
    use_compact_frame_index: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Runs a YOLO model over a sequence and returns detections for tracking.

    Args:
        model (YOLO): Ultralytics YOLO model.
        dataset: YOLOTrackingDataset-like object.
        sequence_name (str): Sequence name.
        conf (float): Detection confidence threshold.
        imgsz (int): YOLO inference image size.
        target_class_id (Optional[int]): Optional class ID filter.
        use_compact_frame_index (bool): If True, frame keys are 1, 2, 3...
            following the extracted sequence order.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary indexed by frame ID.
    """
    detections = {}
    sequence_indices = dataset.get_sequence_indices(sequence_name)

    for compact_frame, idx in enumerate(sequence_indices, start=1):
        image, target = dataset[idx]

        original_frame_number = int(target["frame_number"])
        frame = compact_frame if use_compact_frame_index else original_frame_number
        image_path = target["image_path"]

        predict_kwargs = {
            "source": str(image_path),
            "conf": conf,
            "imgsz": imgsz,
            "verbose": False,
        }

        if target_class_id is not None:
            predict_kwargs["classes"] = [int(target_class_id)]

        _sync_cuda_if_available()
        t0 = perf_counter()

        result = model(**predict_kwargs)[0]

        _sync_cuda_if_available()
        tiempo_detector_sec = perf_counter() - t0

        if result.boxes is None or len(result.boxes) == 0:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            xywh = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
            score = np.zeros((0,), dtype=np.float32)
        else:
            xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            xywh = result.boxes.xywh.detach().cpu().numpy().astype(np.float32)
            cls = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
            score = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

        detections[int(frame)] = {
            "xyxy": xyxy,
            "xywh": xywh,
            "cls": cls,
            "conf": score,
            "image": image,
            "image_path": image_path,
            "original_frame_number": original_frame_number,
            "tiempo_detector_sec": float(tiempo_detector_sec),
        }

    return detections


def save_detections(
    detections: Dict[int, Dict[str, Any]],
    output_path: Path,
    include_images: bool = False,
) -> None:
    """
    Saves a detection dictionary as a .npy file.

    Args:
        detections (Dict[int, Dict[str, Any]]): Detection dictionary indexed by frame.
        output_path (Path): Destination .npy path.
        include_images (bool): If False, removes raw image arrays before saving.

    Returns:
        None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if include_images:
        to_save = detections
    else:
        to_save = {}
        for frame, det in detections.items():
            to_save[frame] = {
                key: value
                for key, value in det.items()
                if key != "image"
            }

    np.save(str(output_path), to_save, allow_pickle=True)


def load_detections(input_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Loads a detection dictionary from a .npy file.

    Args:
        input_path (Path): Path to a .npy detection dictionary.

    Returns:
        Dict[int, Dict[str, Any]]: Loaded detection dictionary.
    """
    return np.load(str(input_path), allow_pickle=True).item()


def _update_tracker(
    tracker,
    detection_results: TrackerDetections,
    frame_image: Optional[np.ndarray] = None,
):
    """
    Updates an Ultralytics tracker with a compatible detection container.

    Args:
        tracker: BYTETracker or BOTSORT instance.
        detection_results (TrackerDetections): Detection object with xyxy, xywh,
            conf and cls attributes.
        frame_image (Optional[np.ndarray]): RGB frame image.

    Returns:
        Any: Raw tracker output.
    """
    try:
        return tracker.update(detection_results, frame_image)
    except TypeError as exc:
        message = str(exc)

        if "positional" in message or "argument" in message or "required" in message:
            return tracker.update(detection_results)

        raise


def run_tracker_from_detections(
    detections: Dict[int, Dict[str, Any]],
    tracker_args: Dict[str, Any],
    return_timings: bool = False,
):
    """
    Runs BYTETracker or BOTSORT using precomputed detections.

    Args:
        detections (Dict[int, Dict[str, Any]]): Dictionary indexed by frame ID.
        tracker_args (Dict[str, Any]): Tracker configuration dictionary.
        return_timings (bool): If True, returns prediction dataframe and timing dataframe.

    Returns:
        pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
            Prediction dataframe, or prediction dataframe and timing dataframe.
    """
    tracker = build_tracker(tracker_args)
    rows = []
    timing_rows = []

    for frame in sorted(detections.keys()):
        det = detections[frame]

        xyxy = np.asarray(det.get("xyxy", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
        xywh = np.asarray(det.get("xywh", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(det.get("conf", np.zeros((0,))), dtype=np.float32).reshape(-1)
        cls = np.asarray(det.get("cls", np.zeros((0,))), dtype=np.float32).reshape(-1)

        detection_results = TrackerDetections(
            xyxy=xyxy,
            xywh=xywh,
            conf=conf,
            cls=cls,
        )

        frame_image = det.get("image", None)
        tiempo_detector_sec = float(det.get("tiempo_detector_sec", 0.0))

        _sync_cuda_if_available()
        t0 = perf_counter()

        tracks = _update_tracker(
            tracker=tracker,
            detection_results=detection_results,
            frame_image=frame_image,
        )

        _sync_cuda_if_available()
        tiempo_tracker_sec = perf_counter() - t0

        timing_rows.append(
            {
                "frame": int(frame),
                "tiempo_detector_sec": float(tiempo_detector_sec),
                "tiempo_tracker_sec": float(tiempo_tracker_sec),
            }
        )

        if tracks is None:
            continue

        tracks = np.asarray(tracks)

        if tracks.size == 0:
            continue

        if tracks.ndim == 1:
            tracks = tracks.reshape(1, -1)

        for tr in tracks:
            if len(tr) < 5:
                continue

            x1 = float(tr[0])
            y1 = float(tr[1])
            x2 = float(tr[2])
            y2 = float(tr[3])
            object_id = int(tr[4])

            score = float(tr[5]) if len(tr) > 5 else 1.0
            class_id = int(tr[6]) if len(tr) > 6 else -1

            bb_width = x2 - x1
            bb_height = y2 - y1

            if bb_width <= 0 or bb_height <= 0:
                continue

            rows.append(
                {
                    "frame": int(frame),
                    "object_id": object_id,
                    "bb_left": x1,
                    "bb_top": y1,
                    "bb_width": bb_width,
                    "bb_height": bb_height,
                    "conf": score,
                    "x": -1.0,
                    "y": -1.0,
                    "z": -1.0,
                    "class_id": class_id,
                    "image_path": det.get("image_path", ""),
                    "original_frame_number": int(det.get("original_frame_number", frame)),
                }
            )

    if not rows:
        pred_df = pd.DataFrame(
            columns=TRACK_COLS
            + [
                "class_id",
                "image_path",
                "original_frame_number",
            ]
        )
    else:
        pred_df = (
            pd.DataFrame(rows)
            .sort_values(["frame", "object_id"])
            .reset_index(drop=True)
        )

    timing_df = pd.DataFrame(timing_rows, columns=TIMING_COLS)

    if return_timings:
        return pred_df, timing_df

    return pred_df


def evaluate_mot_sequence(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    iou_match_threshold: float = 0.5,
    sequence_name: str = "sequence",
) -> pd.DataFrame:
    """
    Evaluates one MOT sequence using motmetrics.

    Args:
        gt_df (pd.DataFrame): Ground-truth dataframe.
        pred_df (pd.DataFrame): Prediction dataframe.
        metrics (Optional[List[str]]): motmetrics metric names.
        iou_match_threshold (float): Minimum IoU required to match GT and prediction.
        sequence_name (str): Name used in the metric summary row.

    Returns:
        pd.DataFrame: motmetrics summary dataframe.
    """
    if metrics is None:
        metrics = DEFAULT_MOT_METRICS

    required_cols = {
        "frame",
        "object_id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
    }

    missing_gt = required_cols - set(gt_df.columns)
    missing_pred = required_cols - set(pred_df.columns)

    if missing_gt:
        raise ValueError(f"gt_df missing columns: {missing_gt}")

    if missing_pred:
        raise ValueError(f"pred_df missing columns: {missing_pred}")

    acc = mm.MOTAccumulator(auto_id=True)

    gt_frames = set(gt_df["frame"].astype(int).tolist()) if not gt_df.empty else set()
    pred_frames = set(pred_df["frame"].astype(int).tolist()) if not pred_df.empty else set()

    all_frames = sorted(gt_frames | pred_frames)
    max_iou_distance = 1.0 - iou_match_threshold

    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_ids = gt_frame["object_id"].astype(int).tolist()
        pred_ids = pred_frame["object_id"].astype(int).tolist()

        gt_boxes = gt_frame[
            ["bb_left", "bb_top", "bb_width", "bb_height"]
        ].to_numpy(dtype=np.float64)

        pred_boxes = pred_frame[
            ["bb_left", "bb_top", "bb_width", "bb_height"]
        ].to_numpy(dtype=np.float64)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            distances = np.empty((len(gt_boxes), len(pred_boxes)), dtype=np.float64)
        else:
            distances = mm.distances.iou_matrix(
                gt_boxes,
                pred_boxes,
                max_iou=max_iou_distance,
            )

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()

    return mh.compute(
        acc,
        metrics=metrics,
        name=sequence_name,
    )


def save_mot_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    cols: Optional[List[str]] = None,
) -> None:
    """
    Saves a MOT-like dataframe without header.

    Args:
        df (pd.DataFrame): Dataframe to save.
        output_path (Path): Output text file.
        cols (Optional[List[str]]): Columns to save. If None, TRACK_COLS is used.

    Returns:
        None
    """
    if cols is None:
        cols = TRACK_COLS

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        output_path.write_text("", encoding="utf-8")
        return

    df[cols].to_csv(
        output_path,
        header=False,
        index=False,
    )


def evaluate_tracker_with_label_detections(
    dataset,
    sequence_name: str,
    tracker_args: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    iou_match_threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Evaluates a tracker using labels_track as detector inputs.

    Args:
        dataset: YOLOTrackingDataset-like object.
        sequence_name (str): Sequence to evaluate.
        tracker_args (Dict[str, Any]): Tracker configuration.
        metrics (Optional[List[str]]): motmetrics metric names.
        iou_match_threshold (float): Minimum IoU required for a valid match.
        output_dir (Optional[Path]): Optional directory to save outputs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
            summary, gt_df, pred_df, detections.
    """
    gt_df = make_gt_mot_dataframe_for_sequence(
        dataset=dataset,
        sequence_name=sequence_name,
        use_compact_frame_index=True,
    )

    detections = get_label_detections_for_sequence(
        dataset=dataset,
        sequence_name=sequence_name,
        use_compact_frame_index=True,
    )

    pred_df, timing_df = run_tracker_from_detections(
        detections=detections,
        tracker_args=tracker_args,
        return_timings=True,
    )

    summary = evaluate_mot_sequence(
        gt_df=gt_df,
        pred_df=pred_df,
        metrics=metrics,
        iou_match_threshold=iou_match_threshold,
        sequence_name=sequence_name,
    )

    summary = _add_timing_to_summary(
        summary=summary,
        timing_df=timing_df,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_mot_dataframe(gt_df, output_dir / "gt.txt")
        save_mot_dataframe(pred_df, output_dir / "pred.txt")
        summary.to_csv(output_dir / "metrics.csv")
        timing_df.to_csv(output_dir / "timings.csv", index=False)
        save_detections(detections, output_dir / "detections.npy", include_images=False)

    return summary, gt_df, pred_df, detections


def evaluate_tracker_with_model_detections(
    model: YOLO,
    dataset,
    sequence_name: str,
    tracker_args: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    target_class_id: Optional[int] = None,
    iou_match_threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Evaluates detector + tracker using YOLO/model detections.

    Args:
        model (YOLO): Ultralytics YOLO model.
        dataset: YOLOTrackingDataset-like object.
        sequence_name (str): Sequence to evaluate.
        tracker_args (Dict[str, Any]): Tracker configuration.
        metrics (Optional[List[str]]): motmetrics metric names.
        conf (float): Detection confidence threshold.
        imgsz (int): YOLO inference image size.
        target_class_id (Optional[int]): Optional class ID filter.
        iou_match_threshold (float): Minimum IoU required for a valid match.
        output_dir (Optional[Path]): Optional directory to save outputs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
            summary, gt_df, pred_df, detections.
    """
    gt_df = make_gt_mot_dataframe_for_sequence(
        dataset=dataset,
        sequence_name=sequence_name,
        use_compact_frame_index=True,
    )

    detections = get_yolo_detections_for_sequence(
        model=model,
        dataset=dataset,
        sequence_name=sequence_name,
        conf=conf,
        imgsz=imgsz,
        target_class_id=target_class_id,
        use_compact_frame_index=True,
    )

    pred_df, timing_df = run_tracker_from_detections(
        detections=detections,
        tracker_args=tracker_args,
        return_timings=True,
    )

    summary = evaluate_mot_sequence(
        gt_df=gt_df,
        pred_df=pred_df,
        metrics=metrics,
        iou_match_threshold=iou_match_threshold,
        sequence_name=sequence_name,
    )

    summary = _add_timing_to_summary(
        summary=summary,
        timing_df=timing_df,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_mot_dataframe(gt_df, output_dir / "gt.txt")
        save_mot_dataframe(pred_df, output_dir / "pred.txt")
        summary.to_csv(output_dir / "metrics.csv")
        timing_df.to_csv(output_dir / "timings.csv", index=False)
        save_detections(detections, output_dir / "detections.npy", include_images=False)

    return summary, gt_df, pred_df, detections


def evaluate_label_detections_on_dataset(
    dataset,
    tracker_args: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    iou_match_threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Evaluates a tracker on all sequences using labels_track as detector inputs.

    Args:
        dataset: YOLOTrackingDataset-like object.
        tracker_args (Dict[str, Any]): Tracker configuration.
        metrics (Optional[List[str]]): motmetrics metric names.
        iou_match_threshold (float): Minimum IoU required for a valid match.
        output_dir (Optional[Path]): Optional base output directory.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
            metrics_summary, gt_by_sequence, pred_by_sequence.
    """
    summaries = []
    gt_by_sequence = {}
    pred_by_sequence = {}

    for sequence_name in dataset.sequence_names:
        print(f"Evaluating GT-label detections: {sequence_name}")

        seq_output_dir = None
        if output_dir is not None:
            seq_output_dir = Path(output_dir) / sequence_name

        summary, gt_df, pred_df, _ = evaluate_tracker_with_label_detections(
            dataset=dataset,
            sequence_name=sequence_name,
            tracker_args=tracker_args,
            metrics=metrics,
            iou_match_threshold=iou_match_threshold,
            output_dir=seq_output_dir,
        )

        summaries.append(summary)
        gt_by_sequence[sequence_name] = gt_df
        pred_by_sequence[sequence_name] = pred_df

    metrics_summary = pd.concat(summaries, axis=0) if summaries else pd.DataFrame()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary.to_csv(output_dir / "metrics_summary.csv")

    return metrics_summary, gt_by_sequence, pred_by_sequence


def evaluate_model_detections_on_dataset(
    model: YOLO,
    dataset,
    tracker_args: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    target_class_id: Optional[int] = None,
    iou_match_threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Evaluates detector + tracker on all dataset sequences.

    Args:
        model (YOLO): Ultralytics YOLO model.
        dataset: YOLOTrackingDataset-like object.
        tracker_args (Dict[str, Any]): Tracker configuration.
        metrics (Optional[List[str]]): motmetrics metric names.
        conf (float): Detection confidence threshold.
        imgsz (int): YOLO inference image size.
        target_class_id (Optional[int]): Optional class ID filter.
        iou_match_threshold (float): Minimum IoU required for a valid match.
        output_dir (Optional[Path]): Optional base output directory.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
            metrics_summary, gt_by_sequence, pred_by_sequence.
    """
    summaries = []
    gt_by_sequence = {}
    pred_by_sequence = {}

    for sequence_name in dataset.sequence_names:
        print(f"Evaluating model detections: {sequence_name}")

        seq_output_dir = None
        if output_dir is not None:
            seq_output_dir = Path(output_dir) / sequence_name

        summary, gt_df, pred_df, _ = evaluate_tracker_with_model_detections(
            model=model,
            dataset=dataset,
            sequence_name=sequence_name,
            tracker_args=tracker_args,
            metrics=metrics,
            conf=conf,
            imgsz=imgsz,
            target_class_id=target_class_id,
            iou_match_threshold=iou_match_threshold,
            output_dir=seq_output_dir,
        )

        summaries.append(summary)
        gt_by_sequence[sequence_name] = gt_df
        pred_by_sequence[sequence_name] = pred_df

    metrics_summary = pd.concat(summaries, axis=0) if summaries else pd.DataFrame()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary.to_csv(output_dir / "metrics_summary.csv")

    return metrics_summary, gt_by_sequence, pred_by_sequence


def render_mot_summary(
    summary: pd.DataFrame,
) -> str:
    """
    Renders a MOTChallenge-like metrics summary.

    Args:
        summary (pd.DataFrame): Summary returned by motmetrics.

    Returns:
        str: Formatted metrics table.
    """
    mh = mm.metrics.create()

    return mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )