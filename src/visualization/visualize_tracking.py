# src/visualization/visualize_tracking.py

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generates a deterministic RGB color for a tracking ID.

    Args:
        track_id (int): Track or object ID.

    Returns:
        Tuple[int, int, int]: RGB color tuple.
    """
    rng = np.random.default_rng(seed=int(track_id) + 12345)
    color = rng.integers(low=40, high=230, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_xyxy_box_with_center(
    image: np.ndarray,
    box_xyxy: Union[List[float], Tuple[float, float, float, float], np.ndarray],
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    line_thickness: int = 2,
    center_radius: int = 4,
    font_scale: float = 0.55,
    draw_label_background: bool = True,
) -> np.ndarray:
    """
    Draws an xyxy bounding box, center point and optional label on an RGB image.

    This function draws in-place on the provided image and returns it.

    Args:
        image (np.ndarray): RGB image.
        box_xyxy (Union[List[float], Tuple[float, float, float, float], np.ndarray]):
            Bounding box in [x_min, y_min, x_max, y_max] format.
        color (Tuple[int, int, int]): RGB color.
        label (Optional[str]): Optional text label.
        line_thickness (int): Bounding box thickness.
        center_radius (int): Radius of the center point.
        font_scale (float): OpenCV font scale.
        draw_label_background (bool): If True, draws a filled background behind
            the label.

    Returns:
        np.ndarray: The same RGB image with drawings applied.
    """
    out = image

    x_min, y_min, x_max, y_max = map(int, box_xyxy)

    cv2.rectangle(
        out,
        (x_min, y_min),
        (x_max, y_max),
        color,
        line_thickness,
    )

    cx = int(round((x_min + x_max) / 2.0))
    cy = int(round((y_min + y_max) / 2.0))

    cv2.circle(
        out,
        (cx, cy),
        radius=center_radius,
        color=color,
        thickness=-1,
    )

    if label is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        text_x = x_min
        text_y = max(15, y_min - 8)

        if draw_label_background:
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            bg_tl = (text_x, text_y - label_h - baseline - 4)
            bg_br = (text_x + label_w + 6, text_y + baseline)

            cv2.rectangle(
                out,
                bg_tl,
                bg_br,
                color,
                -1,
            )

            text_color = (255, 255, 255)
            put_text_point = (text_x + 3, text_y - 3)
        else:
            text_color = color
            put_text_point = (text_x, text_y)

        cv2.putText(
            out,
            label,
            put_text_point,
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return out


def draw_tracking_target(
    image: np.ndarray,
    target: Dict[str, Any],
    class_names: Optional[Dict[int, str]] = None,
    prefix: str = "GT",
    draw_label: bool = True,
) -> np.ndarray:
    """
    Draws ground-truth tracking annotations from a YOLOTrackingDataset target.

    Args:
        image (np.ndarray): RGB image.
        target (Dict[str, Any]): Target returned by YOLOTrackingDataset. Expected keys:
            boxes_xyxy_abs, labels, track_ids.
        class_names (Optional[Dict[int, str]]): Optional mapping from class ID to
            class name.
        prefix (str): Text prefix for labels, usually "GT".
        draw_label (bool): If True, draws text labels.

    Returns:
        np.ndarray: RGB image with GT tracking annotations.
    """
    vis = image.copy()

    boxes = target["boxes_xyxy_abs"]
    labels = target["labels"]
    track_ids = target["track_ids"]

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(track_ids):
        track_ids = track_ids.detach().cpu().numpy()

    for box, class_id, track_id in zip(boxes, labels, track_ids):
        class_id = int(class_id)
        track_id = int(track_id)

        color = get_track_color(track_id)

        if class_names and class_id in class_names:
            class_label = class_names[class_id]
        else:
            class_label = f"cls={class_id}"

        text = f"{prefix} {class_label} id={track_id}" if draw_label else None

        vis = draw_xyxy_box_with_center(
            image=vis,
            box_xyxy=box,
            color=color,
            label=text,
        )

    return vis


def draw_tracking_predictions_from_df(
    image: np.ndarray,
    pred_df: Optional[pd.DataFrame],
    class_names: Optional[Dict[int, str]] = None,
    prefix: str = "PR",
    draw_label: bool = True,
) -> np.ndarray:
    """
    Draws tracking predictions from a MOT-like dataframe.

    Args:
        image (np.ndarray): RGB image.
        pred_df (Optional[pd.DataFrame]): Prediction dataframe. Expected columns:
            object_id, bb_left, bb_top, bb_width, bb_height, conf, class_id.
        class_names (Optional[Dict[int, str]]): Optional mapping from class ID to
            class name.
        prefix (str): Text prefix for labels, usually "PR".
        draw_label (bool): If True, draws text labels.

    Returns:
        np.ndarray: RGB image with prediction annotations.
    """
    vis = image.copy()

    if pred_df is None or pred_df.empty:
        return vis

    for _, row in pred_df.iterrows():
        track_id = int(row["object_id"])
        class_id = int(row["class_id"]) if "class_id" in row else -1
        score = float(row["conf"]) if "conf" in row else 1.0

        x_min = float(row["bb_left"])
        y_min = float(row["bb_top"])
        x_max = x_min + float(row["bb_width"])
        y_max = y_min + float(row["bb_height"])

        color = get_track_color(track_id)

        if class_names and class_id in class_names:
            class_label = class_names[class_id]
        else:
            class_label = f"cls={class_id}"

        text = (
            f"{prefix} {class_label} id={track_id} conf={score:.2f}"
            if draw_label
            else None
        )

        vis = draw_xyxy_box_with_center(
            image=vis,
            box_xyxy=[x_min, y_min, x_max, y_max],
            color=color,
            label=text,
            line_thickness=2,
            center_radius=4,
        )

    return vis


def draw_gt_and_tracking_predictions(
    image: np.ndarray,
    target: Dict[str, Any],
    pred_df: Optional[pd.DataFrame] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """
    Draws GT and model tracking predictions on the same RGB image.

    Args:
        image (np.ndarray): RGB image.
        target (Dict[str, Any]): Target returned by YOLOTrackingDataset.
        pred_df (Optional[pd.DataFrame]): Prediction dataframe for the current frame.
        class_names (Optional[Dict[int, str]]): Optional mapping from class ID to
            class name.

    Returns:
        np.ndarray: RGB image with GT and prediction annotations.
    """
    vis = image.copy()

    vis = draw_tracking_target(
        image=vis,
        target=target,
        class_names=class_names,
        prefix="GT",
        draw_label=True,
    )

    vis = draw_tracking_predictions_from_df(
        image=vis,
        pred_df=pred_df,
        class_names=class_names,
        prefix="PR",
        draw_label=True,
    )

    return vis


def plot_tracking_frame(
    image: np.ndarray,
    target: Dict[str, Any],
    pred_df: Optional[pd.DataFrame] = None,
    class_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    save_pdf: bool = True,
) -> None:
    """
    Plots one tracking frame with GT and optional prediction annotations.

    Args:
        image (np.ndarray): RGB image.
        target (Dict[str, Any]): Target returned by YOLOTrackingDataset.
        pred_df (Optional[pd.DataFrame]): Optional prediction dataframe for the frame.
        class_names (Optional[Dict[int, str]]): Optional mapping from class ID to
            class name.
        title (Optional[str]): Optional plot title.
        save_path (Optional[Union[str, Path]]): Optional path to save the figure.
        save_pdf (bool): If True and save_path is provided, also saves a PDF copy.

    Returns:
        None
    """
    vis = draw_gt_and_tracking_predictions(
        image=image,
        target=target,
        pred_df=pred_df,
        class_names=class_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.imshow(vis)

    if title is None:
        title = f"{target.get('sequence_name', '')} | frame={target.get('frame_number', '')}"

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            str(save_path),
            dpi=150,
            bbox_inches="tight",
        )

        if save_pdf:
            plt.savefig(
                str(save_path.with_suffix(".pdf")),
                format="pdf",
                dpi=150,
                bbox_inches="tight",
            )

    plt.show()
    plt.close(fig)


def mot_dataframe_to_frame_dict(
    pred_df: Optional[pd.DataFrame],
) -> Dict[int, pd.DataFrame]:
    """
    Converts a MOT-like dataframe to a dictionary indexed by frame.

    Args:
        pred_df (Optional[pd.DataFrame]): MOT-like dataframe with a frame column.

    Returns:
        Dict[int, pd.DataFrame]: Dictionary where each key is a frame ID and each
        value is the subset dataframe for that frame.
    """
    if pred_df is None or pred_df.empty:
        return {}

    return {
        int(frame): frame_df.copy()
        for frame, frame_df in pred_df.groupby("frame")
    }


def save_tracking_sequence_frames(
    dataset,
    sequence_name: str,
    output_dir: Union[str, Path],
    pred_by_frame: Optional[Dict[int, pd.DataFrame]] = None,
    class_names: Optional[Dict[int, str]] = None,
    use_compact_frame_index: bool = True,
) -> None:
    """
    Saves all annotated frames of a tracking sequence.

    Args:
        dataset: YOLOTrackingDataset-like object.
        sequence_name (str): Sequence name.
        output_dir (Union[str, Path]): Output directory for annotated images.
        pred_by_frame (Optional[Dict[int, pd.DataFrame]]): Optional dictionary of
            prediction dataframes indexed by frame ID.
        class_names (Optional[Dict[int, str]]): Optional mapping from class ID to
            class name.
        use_compact_frame_index (bool): If True, prediction frames are expected as
            compact frame IDs 1, 2, 3... If False, prediction frames are expected as
            original frame numbers.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_indices = dataset.get_sequence_indices(sequence_name)

    for compact_frame, idx in enumerate(sequence_indices, start=1):
        image, target = dataset[idx]

        if use_compact_frame_index:
            frame_key = compact_frame
        else:
            frame_key = int(target["frame_number"])

        pred_df = None
        if pred_by_frame is not None:
            pred_df = pred_by_frame.get(int(frame_key), None)

        vis = draw_gt_and_tracking_predictions(
            image=image,
            target=target,
            pred_df=pred_df,
            class_names=class_names,
        )

        out_name = Path(target["image_path"]).name
        out_path = output_dir / out_name

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), vis_bgr)


def export_tracking_sequence_video(
    dataset,
    sequence_name: str,
    output_path: Union[str, Path],
    pred_by_frame: Optional[Dict[int, pd.DataFrame]] = None,
    class_names: Optional[Dict[int, str]] = None,
    fps: float = 10.0,
    use_compact_frame_index: bool = True,
) -> None:
    """
    Exports an annotated tracking sequence as an MP4 video.

    If pred_by_frame is None, the video contains only GT tracking annotations.
    If pred_by_frame is provided, the video contains GT and predictions.

    Args:
        dataset: YOLOTrackingDataset-like object.
        sequence_name (str): Sequence name.
        output_path (Union[str, Path]): Output .mp4 path.
        pred_by_frame (Optional[Dict[int, pd.DataFrame]]): Optional dictionary of
            prediction dataframes indexed by frame ID.
        class_names (Optional[Dict[int, str]]): Optional mapping from class ID to
            class name.
        fps (float): Output video FPS.
        use_compact_frame_index (bool): If True, prediction frames are expected as
            compact frame IDs 1, 2, 3... If False, prediction frames are expected as
            original frame numbers.

    Returns:
        None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequence_indices = dataset.get_sequence_indices(sequence_name)

    if len(sequence_indices) == 0:
        raise ValueError(f"No frames found for sequence: {sequence_name}")

    first_image, _ = dataset[sequence_indices[0]]
    height, width = first_image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(fps),
        (width, height),
    )

    try:
        for compact_frame, idx in enumerate(sequence_indices, start=1):
            image, target = dataset[idx]

            if use_compact_frame_index:
                frame_key = compact_frame
            else:
                frame_key = int(target["frame_number"])

            pred_df = None
            if pred_by_frame is not None:
                pred_df = pred_by_frame.get(int(frame_key), None)

            vis = draw_gt_and_tracking_predictions(
                image=image,
                target=target,
                pred_df=pred_df,
                class_names=class_names,
            )

            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            writer.write(vis_bgr)

    finally:
        writer.release()