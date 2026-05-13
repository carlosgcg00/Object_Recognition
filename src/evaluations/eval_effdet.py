# src/evaluations/eval_effdet.py

import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm



def yxyx_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max].

    Args:
        boxes (np.ndarray): Array of boxes with shape (N, 4).

    Returns:
        np.ndarray: Converted boxes with shape (N, 4).
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    return boxes[:, [1, 0, 3, 2]]


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes IoU between two sets of boxes in [x_min, y_min, x_max, y_max] format.

    Args:
        boxes1 (np.ndarray): First set of boxes with shape (N, 4).
        boxes2 (np.ndarray): Second set of boxes with shape (M, 4).

    Returns:
        np.ndarray: IoU matrix with shape (N, M).
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(x11[:, None], x21[None, :])
    inter_y1 = np.maximum(y11[:, None], y21[None, :])
    inter_x2 = np.minimum(x12[:, None], x22[None, :])
    inter_y2 = np.minimum(y12[:, None], y22[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    area2 = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / np.maximum(union, 1e-8)

    return iou.astype(np.float32)


def collect_detection_records(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    """
    Runs inference once and stores predictions and ground-truth annotations on CPU.

    Args:
        model (torch.nn.Module): Detection model.
        dataloader (torch.utils.data.DataLoader): DataLoader to evaluate.
        device (torch.device): Device used for inference.

    Returns:
        List[Dict[str, np.ndarray]]: List of per-image prediction and target records.
    """
    device = torch.device(device)

    model.eval()
    if hasattr(model, "switch_to_predict"):
        model.switch_to_predict()

    records = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Collecting detections", leave=False):
            images_stack = torch.stack(images).to(device).float()
            detections = model(images_stack)

            for i in range(len(images)):
                det = detections[i]

                if det is None or len(det) == 0:
                    pred_boxes = np.zeros((0, 4), dtype=np.float32)
                    pred_scores = np.zeros((0,), dtype=np.float32)
                    pred_labels = np.zeros((0,), dtype=np.int64)
                else:
                    det_np = det.detach().cpu().numpy()
                    pred_boxes = det_np[:, :4].astype(np.float32)
                    pred_scores = det_np[:, 4].astype(np.float32)
                    pred_labels = det_np[:, 5].astype(np.int64)

                gt_boxes_yxyx = targets[i]["bbox"].detach().cpu().numpy().astype(np.float32)
                gt_boxes = yxyx_to_xyxy_np(gt_boxes_yxyx)
                gt_labels = targets[i]["cls"].detach().cpu().numpy().astype(np.int64)

                records.append({
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "pred_labels": pred_labels,
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels,
                })

    return records


def build_detection_confusion_matrix(
    records: List[Dict[str, np.ndarray]],
    class_names: Dict[int, str],
    iou_thresh: float = 0.50,
    score_thresh: float = 0.50,
) -> np.ndarray:
    """
    Builds a detection confusion matrix using greedy IoU matching.

    Matrix convention:
        rows    = ground truth classes
        columns = predicted classes
        index 0 = background

    Args:
        records (List[Dict[str, np.ndarray]]): Stored predictions and targets.
        class_names (Dict[int, str]): Mapping of class IDs to class names.
        iou_thresh (float): Minimum IoU to match a prediction with a ground-truth box.
        score_thresh (float): Minimum confidence score to keep a prediction.

    Returns:
        np.ndarray: Confusion matrix with shape (num_classes + 1, num_classes + 1).
    """
    class_ids = list(class_names.keys())
    class_to_idx = {class_id: idx + 1 for idx, class_id in enumerate(class_ids)}

    num_classes = len(class_ids)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for record in records:
        pred_boxes = record["pred_boxes"]
        pred_scores = record["pred_scores"]
        pred_labels = record["pred_labels"]

        gt_boxes = record["gt_boxes"]
        gt_labels = record["gt_labels"]

        valid_pred_mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[valid_pred_mask]
        pred_scores = pred_scores[valid_pred_mask]
        pred_labels = pred_labels[valid_pred_mask]

        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        gt_matrix_labels = [
            class_to_idx[int(label)]
            for label in gt_labels
            if int(label) in class_to_idx
        ]

        if len(gt_matrix_labels) != len(gt_boxes):
            valid_gt_mask = np.array([int(label) in class_to_idx for label in gt_labels], dtype=bool)
            gt_boxes = gt_boxes[valid_gt_mask]

        matched_gt = set()

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            pred_label = int(pred_label)

            if pred_label not in class_to_idx:
                continue

            pred_col = class_to_idx[pred_label]

            if len(gt_boxes) == 0:
                cm[0, pred_col] += 1
                continue

            ious = box_iou_np(pred_box.reshape(1, 4), gt_boxes)[0]

            for gt_idx in matched_gt:
                ious[gt_idx] = -1.0

            best_gt_idx = int(np.argmax(ious))
            best_iou = float(ious[best_gt_idx])

            if best_iou >= iou_thresh:
                gt_row = gt_matrix_labels[best_gt_idx]
                cm[gt_row, pred_col] += 1
                matched_gt.add(best_gt_idx)
            else:
                cm[0, pred_col] += 1

        for gt_idx, gt_row in enumerate(gt_matrix_labels):
            if gt_idx not in matched_gt:
                cm[gt_row, 0] += 1

    return cm


def summarize_confusion_matrix(
    cm: np.ndarray,
    class_names: Dict[int, str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Computes precision, recall and F1 from a detection confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (Dict[int, str]): Mapping of class IDs to names.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]:
            - Per-class metrics.
            - Global micro/macro metrics.
    """
    rows = []
    class_ids = list(class_names.keys())

    for class_idx, class_id in enumerate(class_ids, start=1):
        class_name = class_names[class_id]

        tp = cm[class_idx, class_idx]
        pred_total = cm[:, class_idx].sum()
        gt_total = cm[class_idx, :].sum()

        fp = pred_total - tp
        fn = gt_total - tp

        precision = tp / pred_total if pred_total > 0 else np.nan
        recall = tp / gt_total if gt_total > 0 else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else np.nan
        )

        rows.append({
            "class_id": int(class_id),
            "class_name": class_name,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "support_gt": int(gt_total),
            "support_pred": int(pred_total),
            "precision": float(precision) if np.isfinite(precision) else np.nan,
            "recall": float(recall) if np.isfinite(recall) else np.nan,
            "f1": float(f1) if np.isfinite(f1) else np.nan,
        })

    per_class_df = pd.DataFrame(rows)

    tp_micro = np.trace(cm[1:, 1:])
    pred_total_micro = cm[:, 1:].sum()
    gt_total_micro = cm[1:, :].sum()

    fp_micro = pred_total_micro - tp_micro
    fn_micro = gt_total_micro - tp_micro

    precision_micro = tp_micro / pred_total_micro if pred_total_micro > 0 else np.nan
    recall_micro = tp_micro / gt_total_micro if gt_total_micro > 0 else np.nan
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if np.isfinite(precision_micro) and np.isfinite(recall_micro) and (precision_micro + recall_micro) > 0
        else np.nan
    )

    global_metrics = {
        "tp": int(tp_micro),
        "fp": int(fp_micro),
        "fn": int(fn_micro),
        "precision_micro": float(precision_micro) if np.isfinite(precision_micro) else np.nan,
        "recall_micro": float(recall_micro) if np.isfinite(recall_micro) else np.nan,
        "f1_micro": float(f1_micro) if np.isfinite(f1_micro) else np.nan,
        "precision_macro": float(per_class_df["precision"].mean(skipna=True)),
        "recall_macro": float(per_class_df["recall"].mean(skipna=True)),
        "f1_macro": float(per_class_df["f1"].mean(skipna=True)),
    }

    return per_class_df, global_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_path: Path,
    normalize: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    Saves a confusion matrix plot.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (List[str]): Axis labels.
        out_path (Path): Output path.
        normalize (bool): If True, normalize by ground-truth rows.
        title (Optional[str]): Plot title.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(np.float32) / np.maximum(row_sums, 1)
        fmt = ".2f"
        default_title = "Normalized Detection Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        default_title = "Detection Confusion Matrix"

    fig_size = max(8, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title or default_title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.0

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            value = cm_plot[i, j]
            text = format(value, fmt)
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_pr_curves(
    records: List[Dict[str, np.ndarray]],
    class_names: Dict[int, str],
    iou_thresh: float = 0.50,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes precision-recall values across confidence thresholds.

    Args:
        records (List[Dict[str, np.ndarray]]): Stored predictions and targets.
        class_names (Dict[int, str]): Mapping of class IDs to names.
        iou_thresh (float): Minimum IoU used for matching.
        thresholds (Optional[np.ndarray]): Confidence thresholds.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Micro-average PR curve.
            - Per-class PR curve.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    micro_rows = []
    class_rows = []

    for score_thresh in thresholds:
        cm = build_detection_confusion_matrix(
            records=records,
            class_names=class_names,
            iou_thresh=iou_thresh,
            score_thresh=float(score_thresh),
        )

        per_class_df, global_metrics = summarize_confusion_matrix(cm, class_names)

        micro_rows.append({
            "score_thresh": float(score_thresh),
            "precision": global_metrics["precision_micro"],
            "recall": global_metrics["recall_micro"],
            "f1": global_metrics["f1_micro"],
        })

        for _, row in per_class_df.iterrows():
            class_rows.append({
                "score_thresh": float(score_thresh),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
            })

    return pd.DataFrame(micro_rows), pd.DataFrame(class_rows)


def plot_precision_recall_curves(
    micro_df: pd.DataFrame,
    class_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Saves precision-recall curves.

    Args:
        micro_df (pd.DataFrame): Micro-average PR curve.
        class_df (pd.DataFrame): Per-class PR curve.
        out_path (Path): Output path.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    ax.plot(
        micro_df["recall"],
        micro_df["precision"],
        linewidth=3,
        label="micro-average",
        color="black",
    )

    for class_name, df_cls in class_df.groupby("class_name"):
        ax.plot(
            df_cls["recall"],
            df_cls["precision"],
            linewidth=2,
            label=str(class_name),
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_vs_threshold(
    micro_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Saves micro-average precision, recall and F1 as a function of confidence threshold.

    Args:
        micro_df (pd.DataFrame): Micro-average metrics across thresholds.
        out_path (Path): Output path.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(micro_df["score_thresh"], micro_df["precision"], label="Precision")
    ax.plot(micro_df["score_thresh"], micro_df["recall"], label="Recall")
    ax.plot(micro_df["score_thresh"], micro_df["f1"], label="F1")

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Precision / Recall / F1 vs Confidence Threshold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_json_safe(obj):
    """
    Converts NaN values into None so the resulting dictionary is JSON-safe.

    Args:
        obj: Object to sanitize.

    Returns:
        Sanitized object.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value

    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj

    return obj


def resolve_detection_source(source, yaml_dir: Path) -> List[str]:
    """
    Resolves image paths from a YOLO dataset split source.

    Args:
        source: Source from the dataset YAML. Can be a txt file, directory or list.
        yaml_dir (Path): Directory containing the dataset YAML.

    Returns:
        List[str]: List of image paths.
    """
    if source is None:
        return []

    if isinstance(source, list):
        paths = []
        for item in source:
            paths.extend(resolve_detection_source(item, yaml_dir))
        return paths

    source_path = Path(source)

    if not source_path.is_absolute():
        source_path = (yaml_dir / source_path).resolve()

    if source_path.is_file() and source_path.suffix.lower() == ".txt":
        with open(source_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        resolved_lines = []
        for line in lines:
            line_path = Path(line)
            if not line_path.is_absolute():
                line_path = (source_path.parent / line_path).resolve()
            resolved_lines.append(str(line_path))

        return resolved_lines

    if source_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([
            str(p)
            for p in source_path.rglob("*")
            if p.suffix.lower() in valid_exts
        ])

    return []


def image_to_yolo_label_path(img_path: Path) -> Path:
    """
    Converts an image path into its corresponding YOLO label path.

    Example:
        dataset/processed/images/horse_1/img.jpg
        -> dataset/processed/labels/horse_1/img.txt

    Args:
        img_path (Path): Image path.

    Returns:
        Path: Label path.
    """
    img_path = Path(img_path)
    parts = list(img_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    # Fallback
    return img_path.with_suffix(".txt")


def read_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a YOLO .txt label file and returns absolute xyxy boxes and class labels.

    Args:
        label_path (Path): Path to YOLO label txt.
        img_w (int): Image width.
        img_h (int): Image height.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - boxes in xyxy absolute pixel format.
            - labels as class IDs.
    """
    label_path = Path(label_path)

    if not label_path.exists():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64)
        )

    boxes = []
    labels = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:5])

            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h

            x_min = x_c - w / 2.0
            y_min = y_c - h / 2.0
            x_max = x_c + w / 2.0
            y_max = y_c + h / 2.0

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cls_id)

    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 4),
        np.array(labels, dtype=np.int64)
    )


def collect_yolo_detection_records(
    model,
    yaml_path: str,
    split: str,
    img_size: int,
    device: str,
) -> List[Dict[str, np.ndarray]]:
    """
    Runs YOLO inference and stores predictions and ground truth in the same
    record format used by EfficientDet diagnostics.

    Args:
        model: Ultralytics YOLO model.
        yaml_path (str): Path to YOLO dataset YAML.
        split (str): Dataset split to evaluate, e.g. 'test' or 'val'.
        img_size (int): Inference image size.
        device (str): Device.

    Returns:
        List[Dict[str, np.ndarray]]: List of detection records.
    """
    yaml_path = Path(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = yaml_path.parent
    source = data_cfg.get(split)
    image_paths = resolve_detection_source(source, yaml_dir)

    records = []

    if len(image_paths) == 0:
        print(f"⚠️ Warning: no images found for YOLO split='{split}' in {yaml_path}")
        return records

    results = model.predict(
        source=image_paths,
        conf=0.001,
        iou=0.70,
        imgsz=img_size,
        save=False,
        verbose=False,
        device=device,
        stream=True
    )

    for img_path_str, result in tqdm(
        zip(image_paths, results),
        total=len(image_paths),
        desc=f"Collecting YOLO {split} detections",
        leave=False
    ):
        img_path = Path(img_path_str)

        orig_h, orig_w = result.orig_shape

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_labels = np.zeros((0,), dtype=np.int64)
        else:
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            pred_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            pred_labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)

        label_path = image_to_yolo_label_path(img_path)
        gt_boxes, gt_labels = read_yolo_label_file(
            label_path=label_path,
            img_w=orig_w,
            img_h=orig_h
        )

        records.append({
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        })

    return records


def evaluate_detection_diagnostics_from_records(
    records: List[Dict[str, np.ndarray]],
    class_names: Dict[int, str],
    output_dir: Path,
    iou_thresh: float = 0.50,
    score_thresh: float = 0.50,
    curve_thresholds: Optional[np.ndarray] = None,
) -> Dict:
    """
    Computes detection diagnostics from already collected prediction/GT records.

    This function is model-agnostic. It can be used for EfficientDet, YOLO or any
    detector if records follow the expected format.

    Args:
        records (List[Dict[str, np.ndarray]]): Detection records.
        class_names (Dict[int, str]): Ordered class mapping.
        output_dir (Path): Output directory.
        iou_thresh (float): IoU threshold for matching.
        score_thresh (float): Confidence threshold for confusion matrix.
        curve_thresholds (Optional[np.ndarray]): Confidence thresholds for PR curves.

    Returns:
        Dict: Diagnostic summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["background"] + list(class_names.values())

    cm = build_detection_confusion_matrix(
        records=records,
        class_names=class_names,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix_counts.csv")

    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(output_dir / "confusion_matrix_normalized_by_gt.csv")

    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        out_path=output_dir / "confusion_matrix_counts.pdf",
        normalize=False,
        title=f"Detection Confusion Matrix | IoU={iou_thresh:.2f}, score={score_thresh:.2f}",
    )

    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        out_path=output_dir / "confusion_matrix_counts.jpg",
        normalize=False,
        title=f"Detection Confusion Matrix | IoU={iou_thresh:.2f}, score={score_thresh:.2f}",
    )

    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        out_path=output_dir / "confusion_matrix_normalized_by_gt.pdf",
        normalize=True,
        title=f"Normalized Detection Confusion Matrix | IoU={iou_thresh:.2f}, score={score_thresh:.2f}",
    )

    plot_confusion_matrix(
        cm=cm,
        labels=labels,
        out_path=output_dir / "confusion_matrix_normalized_by_gt.jpg",
        normalize=True,
        title=f"Normalized Detection Confusion Matrix | IoU={iou_thresh:.2f}, score={score_thresh:.2f}",
    )

    per_class_df, global_metrics = summarize_confusion_matrix(
        cm=cm,
        class_names=class_names
    )

    per_class_df.to_csv(output_dir / "precision_recall_f1_per_class.csv", index=False)

    with open(output_dir / "precision_recall_f1_global.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(global_metrics), f, indent=4)

    micro_curve_df, class_curve_df = compute_pr_curves(
        records=records,
        class_names=class_names,
        iou_thresh=iou_thresh,
        thresholds=curve_thresholds,
    )

    micro_curve_df.to_csv(output_dir / "pr_curve_micro.csv", index=False)
    class_curve_df.to_csv(output_dir / "pr_curve_per_class.csv", index=False)

    plot_precision_recall_curves(
        micro_df=micro_curve_df,
        class_df=class_curve_df,
        out_path=output_dir / "precision_recall_curves.pdf",
    )

    plot_precision_recall_vs_threshold(
        micro_df=micro_curve_df,
        out_path=output_dir / "precision_recall_f1_vs_score_threshold.pdf",
    )

    result = {
        "iou_thresh": float(iou_thresh),
        "score_thresh": float(score_thresh),
        "num_images": len(records),
        "output_dir": str(output_dir),
        "global_metrics": global_metrics,
        "per_class_metrics": per_class_df.to_dict(orient="records"),
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_labels": labels,
        "files": {
            "confusion_matrix_counts_csv": str(output_dir / "confusion_matrix_counts.csv"),
            "confusion_matrix_normalized_csv": str(output_dir / "confusion_matrix_normalized_by_gt.csv"),
            "confusion_matrix_counts_pdf": str(output_dir / "confusion_matrix_counts.pdf"),
            "confusion_matrix_counts_jpg": str(output_dir / "confusion_matrix_counts.jpg"),
            "confusion_matrix_normalized_pdf": str(output_dir / "confusion_matrix_normalized_by_gt.pdf"),
            "confusion_matrix_normalized_jpg": str(output_dir / "confusion_matrix_normalized_by_gt.jpg"),
            "per_class_metrics_csv": str(output_dir / "precision_recall_f1_per_class.csv"),
            "global_metrics_json": str(output_dir / "precision_recall_f1_global.json"),
            "pr_curve_micro_csv": str(output_dir / "pr_curve_micro.csv"),
            "pr_curve_per_class_csv": str(output_dir / "pr_curve_per_class.csv"),
            "pr_curves_pdf": str(output_dir / "precision_recall_curves.pdf"),
            "precision_recall_f1_vs_threshold_pdf": str(output_dir / "precision_recall_f1_vs_score_threshold.pdf"),
        },
    }

    with open(output_dir / "detection_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(result), f, indent=4)

    return make_json_safe(result)


def evaluate_detection_diagnostics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Dict[int, str],
    output_dir: Path,
    iou_thresh: float = 0.50,
    score_thresh: float = 0.50,
    curve_thresholds: Optional[np.ndarray] = None,
) -> Dict:
    """
    Computes and saves detection diagnostics for EfficientDet.

    Args:
        model (torch.nn.Module): Detection model.
        dataloader (torch.utils.data.DataLoader): Evaluation DataLoader.
        device (torch.device): Device used for inference.
        class_names (Dict[int, str]): Ordered class mapping.
        output_dir (Path): Directory where diagnostics will be saved.
        iou_thresh (float): IoU threshold used for matching.
        score_thresh (float): Confidence threshold used for the confusion matrix.
        curve_thresholds (Optional[np.ndarray]): Thresholds used for PR curves.

    Returns:
        Dict: JSON-serializable summary of the diagnostic results.
    """
    records = collect_detection_records(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    return evaluate_detection_diagnostics_from_records(
        records=records,
        class_names=class_names,
        output_dir=output_dir,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
        curve_thresholds=curve_thresholds,
    )