# src/dataset/tracking_dataset.py

import re
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional, Dict, List, Tuple, Any

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class YOLOTrackingDataset(Dataset):
    """
    Dataset for tracking tasks with YOLO format annotations.
    
    The target format is a dictionary containing:
        class_id, target_id, x_min, y_min, x_max, y_max
    """
    
    def __init__(
        self,
        split_txt: Path,
        image_root_name: str = "images_track",
        label_root_name: str = "labels_track",
        rgb: bool = True,
        return_tensors: bool = True,
        seed: int = 42,
    ):
        self.split_txt = Path(split_txt)
        self.image_root_name = image_root_name
        self.label_root_name = label_root_name
        self.rgb = rgb
        self.return_tensors = return_tensors
        self.seed = seed

        # Extract the names of the image
        self.img_paths = self._load_imgs(self.split_txt)
        # Sort the image paths, as we have for example: pig_1_frame_5.jpg, pig_1_frame_10.jpg, 
        # pig_1_frame_15.jpg, etc. We want to ensure that the frame order is correct.
        self.img_paths = sorted(self.img_paths, key=self._natural_path_key)

        self.sequence_to_indices = self._build_sequence_index()
        self.sequence_names = sorted(self.sequence_to_indices.keys())

    def _load_imgs(self, split_txt: Path) -> List[Path]:
        if not split_txt.exists():
            raise FileNotFoundError(f"split_txt not found: {split_txt}")

        with open(split_txt, "r", encoding="utf-8") as f:
            img_paths = [Path(line.strip()) for line in f.readlines() if line.strip()]

        if len(img_paths) == 0:
            raise ValueError(f"No image paths found in: {split_txt}")

        return img_paths

    @staticmethod
    def _natural_path_key(path: Path):
        """
        Generates a key for natural sorting of file paths, 
        ensuring that numeric parts are sorted as numbers.
        """
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", str(path))
        ]

    @staticmethod
    def _extract_frame_number(path: Path) -> int:
        """
        Extracts the frame number from a path.
        
        Image name: pig_1_frame_5.jpg -> frame number: 5
        
        Args:
            path (Path): Path to the image file.
        Returns:
            frame_number (int): Frame number.
        """
        numbers = re.findall(r"\d+", path.stem)
        if len(numbers) == 0:
            return -1
        return int(numbers[-1])
    
    def _resolve_label_path(self, img_path: Path) -> Path:
        """
        Transforms an image path to its corresponding label path.
        
        Example:
            From: img_path: /data/images/pig_1_frame_5.jpg
            To: label_path: /data/labels_track/pig_1_frame_5.txt
            
        Args:
            img_path (Path): Path to the image file.
        Returns:
            label_path (Path): Path to the corresponding label file (label_track path).        
        """
        img_parts = list(img_path.parts)

        if self.image_root_name in img_parts:
            idx = img_parts.index(self.image_root_name)
            img_parts[idx] = self.label_root_name
            return Path(*img_parts).with_suffix(".txt")

        return (
            img_path.parent.parent
            / self.label_root_name
            / img_path.parent.name
            / img_path.with_suffix(".txt").name
        )
        
    def _build_sequence_index(self) -> Dict[str, List[int]]:
        """
        Builds an index mapping sequence names to lists of frame indices.
        Returns:
            Dict[str, List[int]]: A dictionary where keys are sequence names and values are lists of frame indices.
        """
        sequence_to_indices = defaultdict(list)

        for idx, img_path in enumerate(self.img_paths):
            sequence_name = img_path.parent.name
            sequence_to_indices[sequence_name].append(idx)

        for sequence_name in sequence_to_indices:
            sequence_to_indices[sequence_name] = sorted(
                sequence_to_indices[sequence_name],
                key=lambda i: self._extract_frame_number(self.img_paths[i]),
            )

        return dict(sequence_to_indices)

    @staticmethod
    def _clip01(value: float) -> float:
        """
        Clips a value to the range [0, 1].
        Assures that the value is a valid normalized coordinate.
        
        Args:
            value (float): Value to be clipped.
        Returns:
            float: Clipped value.
        """
        return max(0.0, min(1.0, float(value)))
    
    def _read_yolo_track_label_file(
        self,
        label_path: Path,
        image_width: int,
        image_height: int,
    ) -> Dict[str, np.ndarray]:
        """
        Reads a YOLO tracking label file and returns a dictionary with various box formats and associated labels and track IDs.
        
        Args:
            label_path (Path): Path to the YOLO tracking label file.
            image_width (int): Width of the image.
            image_height (int): Height of the image.
        Returns:
            Dict[str, np.ndarray]: A dictionary containing:
                - "boxes_yolo": (N, 4) array of [x_center, y_center, width, height] normalized to [0, 1].
                - "boxes_xyxy_norm": (N, 4) array of [x_min, y_min, x_max, y_max] normalized to [0, 1].
                - "boxes_xyxy_abs": (N, 4) array of [x_min, y_min, x_max, y_max] in absolute pixels.
                - "boxes_xywh_abs": (N, 4) array of [x_min, y_min, width, height] in absolute pixels.
                - "labels": (N,) array of class labels.
                - "track_ids": (N,) array of track IDs.
        """
        
        
        boxes_yolo = []
        boxes_xyxy_norm = []
        boxes_xyxy_abs = []
        boxes_xywh_abs = []
        labels = []
        track_ids = []

        if not label_path.exists():
            return {
                "boxes_yolo": np.zeros((0, 4), dtype=np.float32),
                "boxes_xyxy_norm": np.zeros((0, 4), dtype=np.float32),
                "boxes_xyxy_abs": np.zeros((0, 4), dtype=np.float32),
                "boxes_xywh_abs": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "track_ids": np.zeros((0,), dtype=np.int64),
            }

        fallback_track_id = 1

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) == 6:
                    class_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    x_c, y_c, w, h = map(float, parts[2:6])

                elif len(parts) == 5:
                    class_id = int(float(parts[0]))
                    track_id = fallback_track_id
                    x_c, y_c, w, h = map(float, parts[1:5])

                else:
                    raise ValueError(
                        f"Formato no soportado en {label_path}: {line}. "
                        "Esperado: 'class_id target_id xc yc w h'."
                    )



                w = abs(float(w))
                h = abs(float(h))

                if w <= 1e-6 or h <= 1e-6:
                    continue

                x_min_n = self._clip01(x_c - w / 2.0)
                y_min_n = self._clip01(y_c - h / 2.0)
                x_max_n = self._clip01(x_c + w / 2.0)
                y_max_n = self._clip01(y_c + h / 2.0)

                if x_max_n <= x_min_n or y_max_n <= y_min_n:
                    continue

                x_min_abs = x_min_n * image_width
                y_min_abs = y_min_n * image_height
                x_max_abs = x_max_n * image_width
                y_max_abs = y_max_n * image_height

                box_w_abs = x_max_abs - x_min_abs
                box_h_abs = y_max_abs - y_min_abs

                boxes_yolo.append([x_c, y_c, w, h])
                boxes_xyxy_norm.append([x_min_n, y_min_n, x_max_n, y_max_n])
                boxes_xyxy_abs.append([x_min_abs, y_min_abs, x_max_abs, y_max_abs])
                boxes_xywh_abs.append([x_min_abs, y_min_abs, box_w_abs, box_h_abs])
                labels.append(class_id)
                track_ids.append(track_id)

                fallback_track_id += 1

        return {
            "boxes_yolo": np.asarray(boxes_yolo, dtype=np.float32).reshape(-1, 4),
            "boxes_xyxy_norm": np.asarray(boxes_xyxy_norm, dtype=np.float32).reshape(-1, 4),
            "boxes_xyxy_abs": np.asarray(boxes_xyxy_abs, dtype=np.float32).reshape(-1, 4),
            "boxes_xywh_abs": np.asarray(boxes_xywh_abs, dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, dtype=np.int64),
            "track_ids": np.asarray(track_ids, dtype=np.int64),
        }


    @staticmethod
    def _resize_boxes(
        boxes_xyxy_abs: np.ndarray,
        boxes_xywh_abs: np.ndarray,
        old_width: int,
        old_height: int,
        new_width: int,
        new_height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(boxes_xyxy_abs) == 0:
            return boxes_xyxy_abs, boxes_xywh_abs

        scale_x = new_width / old_width
        scale_y = new_height / old_height

        boxes_xyxy_abs = boxes_xyxy_abs.copy()
        boxes_xyxy_abs[:, [0, 2]] *= scale_x
        boxes_xyxy_abs[:, [1, 3]] *= scale_y

        boxes_xywh_abs = boxes_xywh_abs.copy()
        boxes_xywh_abs[:, [0, 2]] *= scale_x
        boxes_xywh_abs[:, [1, 3]] *= scale_y

        return boxes_xyxy_abs, boxes_xywh_abs

    def _to_tensor_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        for key in [
            "boxes_yolo",
            "boxes_xyxy_norm",
            "boxes_xyxy_abs",
            "boxes_xywh_abs",
        ]:
            target[key] = torch.as_tensor(target[key], dtype=torch.float32)

        for key in ["labels", "track_ids"]:
            target[key] = torch.as_tensor(target[key], dtype=torch.int64)

        target["original_size"] = torch.as_tensor(target["original_size"], dtype=torch.int64)
        target["image_size"] = torch.as_tensor(target["image_size"], dtype=torch.int64)

        return target

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        label_path = self._resolve_label_path(img_path)

        image = cv2.imread(str(img_path))

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        original_height, original_width = image.shape[:2]

        label_data = self._read_yolo_track_label_file(
            label_path=label_path,
            image_width=original_width,
            image_height=original_height,
        )

        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        sequence_name = img_path.parent.name
        frame_number = self._extract_frame_number(img_path)

        target = {
            "boxes_yolo": label_data["boxes_yolo"],
            "boxes_xyxy_norm": label_data["boxes_xyxy_norm"],
            "boxes_xyxy_abs": label_data["boxes_xyxy_abs"],
            "boxes_xywh_abs": label_data["boxes_xywh_abs"],
            "labels": label_data["labels"],
            "track_ids": label_data["track_ids"],
            "image_path": str(img_path),
            "label_path": str(label_path),
            "sequence_name": sequence_name,
            "frame_index": idx,
            "frame_number": frame_number,
            "original_size": np.asarray([original_height, original_width], dtype=np.int64),
            "image_size": np.asarray([image.shape[0], image.shape[1]], dtype=np.int64),
        }

        if self.return_tensors:
            target = self._to_tensor_target(target)

        return image, target

    def get_sequence_indices(self, sequence_name: str) -> List[int]:
        if sequence_name not in self.sequence_to_indices:
            raise KeyError(
                f"Sequence '{sequence_name}' not found. "
                f"Available sequences: {self.sequence_names}"
            )

        return self.sequence_to_indices[sequence_name]

    def iter_sequence(self, sequence_name: str):
        for idx in self.get_sequence_indices(sequence_name):
            yield self[idx]

    def get_sequence_items(self, sequence_name: str):
        return [self[idx] for idx in self.get_sequence_indices(sequence_name)]

    def summary(self) -> Dict[str, Any]:
        return {
            "split_txt": str(self.split_txt),
            "num_images": len(self.img_paths),
            "num_sequences": len(self.sequence_names),
            "sequence_names": self.sequence_names,
            "num_frames_per_sequence": {
                seq: len(indices)
                for seq, indices in self.sequence_to_indices.items()
            },
            "label_root_name": self.label_root_name,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        for idx in range(len(self)):
            _, target = self[idx]

            boxes_xyxy_abs = target["boxes_xyxy_abs"]
            boxes_xywh_abs = target["boxes_xywh_abs"]
            labels = target["labels"]
            track_ids = target["track_ids"]

            if torch.is_tensor(boxes_xyxy_abs):
                boxes_xyxy_abs = boxes_xyxy_abs.cpu().numpy()
            if torch.is_tensor(boxes_xywh_abs):
                boxes_xywh_abs = boxes_xywh_abs.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(track_ids):
                track_ids = track_ids.cpu().numpy()

            for box_xyxy, box_xywh, class_id, track_id in zip(
                boxes_xyxy_abs,
                boxes_xywh_abs,
                labels,
                track_ids,
            ):
                x_min, y_min, x_max, y_max = box_xyxy
                x, y, w, h = box_xywh

                rows.append(
                    {
                        "sequence_name": target["sequence_name"],
                        "frame_index": int(target["frame_index"]),
                        "frame_number": int(target["frame_number"]),
                        "image_path": target["image_path"],
                        "label_path": target["label_path"],
                        "class_id": int(class_id),
                        "track_id": int(track_id),
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max),
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                    }
                )

        return pd.DataFrame(rows)