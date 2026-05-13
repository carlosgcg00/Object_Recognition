# src/engine/train_yolo.py

import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import json
import yaml
import time
import os
import shutil
import datetime
import gc
import numpy as np

from utils.file_utils import read_yaml, build_ordered_class_dict
from visualization.visualize import plot_yolo_curves
from evaluations.eval_effdet import (
    collect_yolo_detection_records,
    evaluate_detection_diagnostics_from_records,
    resolve_detection_source
)

# ==========================================
# UTILITIES
# ==========================================

def get_yolo_params_info(yolo_model):
    """
    Extracts parameter information directly from the internal Torch model
    ensuring it reflects the current 'requires_grad' state.

    Args:
        yolo_model: YOLO model to extract parameter information from.
    
    Returns:
        dict: Dictionary containing parameter information.
    """
    inner_model = yolo_model.model
    total_params = sum(p.numel() for p in inner_model.parameters())
    trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_percentage": round((trainable_params / total_params) * 100, 2)
    }
    

def profile_yolo_inference(model, img_size, device, num_warmup=10, num_rep=100):
    """
    Measures YOLO inference latency matching EfficientDet logic.

    Args:
        model: YOLO model to profile.
        img_size (int): Input image size.
        device (str): Device to use for inference.
        num_warmup (int): Number of warmup iterations.
        num_rep (int): Number of repetition iterations.
    
    Returns:
        dict: Dictionary containing inference statistics.
    """
    model.to(device)
    dummy = torch.rand(1, 3, img_size, img_size).to(device)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    print(f"⏱️  Profiling inference latency on {device}...")
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(dummy)
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_rep):
            if device.type == 'cuda':
                starter.record()
                _ = model(dummy)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000.0 
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
                curr_time = time.perf_counter() - t0

            latencies.append(curr_time)
            
    avg_batch_time = np.mean(latencies)

    return {
        'batch_size': 1,
        'mean_batch_time_sec': avg_batch_time,
        'std_batch_time_sec': np.std(latencies),
        'mean_img_time_sec': avg_batch_time,
        'fps_batch_based': 1.0 / avg_batch_time
    }


# ==========================================
# PRINCIPAL FUNCTION
# ==========================================

def train_yolo_pipeline(
    architecture: str,
    yaml_path: str,
    output_dir: str,
    epochs_tl: int = 10,
    epochs_ft: int = 20,
    batch_size: int = 16,
    img_size: int = 640,
    lr_tl: float = 1e-3,
    lr_ft: float = 1e-4,
    optimizer_str: str = 'AdamW', 
    weight_decay: float = 0.0005,
    dropout_rate: float = 0.1,
    warmup_epochs: int = 3,
    early_stop_patience: int = 5,
    mosaic_prob: float = 1.0, 
    device: str = 'cuda',
    class_order: list = None
):
    """
    Advanced YOLO pipeline aligned with EfficientDet metrics and reporting.

    Args:
        architecture (str): YOLO model architecture (e.g., 'yolov11n', 'yolov11s').
        yaml_path (str): Path to the dataset YAML configuration.
        output_dir (str): Directory to save results.
        epochs_tl (int): Number of epochs for Transfer Learning (Phase 1).
        epochs_ft (int): Number of epochs for Fine Tuning (Phase 2).
        batch_size (int): Batch size for training.
        img_size (int): Input image size.
        lr_tl (float): Learning rate for Transfer Learning.
        lr_ft (float): Learning rate for Fine Tuning.
        optimizer_str (str): Optimizer to use (e.g., 'AdamW', 'SGD').
        weight_decay (float): Weight decay for regularization.
        dropout_rate (float): Dropout rate for regularization.
        early_stop_patience (int): Patience for early stopping based on validation performance.
        warmup_epochs (int): Number of epochs to warm up the learning rate.
        mosaic_prob (float): Probability of using mosaic augmentation.
        device (str): Device to use for training ('cuda' or 'cpu').
        class_order (list): Optional list specifying the desired class order for reporting.
    
    Returns:
        dict: Dictionary containing training results.
    """
    start_time = datetime.datetime.now()
    final_out = Path(output_dir)
    final_out.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    # 1. Data Configuration (to get class names)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    class_names = build_ordered_class_dict(
        dataset_names=data_cfg['names'],
        class_order=class_order if class_order is not None else data_cfg['names'],
        one_based=False
    )

    # --- PHASE 1: TRANSFER LEARNING (Backbone Frozen) ---
    print("\n" + "="*60 + f"\nPhase 1: Transfer Learning (TL) - {epochs_tl} epochs\n" + "="*60)
    model = YOLO(architecture)
    
    # Manually freeze for Phase 1 reporting
    # freeze=23 freezes the first 10 layers (usually the backbone)
    freeze_count = 23
    for i, (name, param) in enumerate(model.model.named_parameters()):
        # Approximate backbone freezing by index for the report
        if i < freeze_count * 3: # Multiplying by 3 as a heuristic for weight/bias/other param split
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Get model info BEFORE training starts
    model_info_tl = get_yolo_params_info(model)
    print(f"🛠️  TL Params: {model_info_tl['trainable_params']:,} trainable / {model_info_tl['total_params']:,} total")

    results_tl = model.train(
        data=yaml_path,
        epochs=epochs_tl,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr_tl,
        optimizer=optimizer_str,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        patience=early_stop_patience,
        cos_lr=False,
        dropout=dropout_rate,
        mosaic=mosaic_prob,
        freeze=freeze_count, 
        project=str(final_out),
        name="phase_tl",
        device=device,
        exist_ok=True,
        save=True,
        plots=True
    )

    # --- PHASE 2: FINE TUNING (Everything Unfrozen) ---
    print("\n" + "="*60 + f"\nPhase 2: Fine Tuning (FT) - {epochs_ft} epochs\n" + "="*60)
    best_tl_weights = Path(results_tl.save_dir) / 'weights' / 'best.pt'
    model = YOLO(best_tl_weights) 
    
    # Explicitly unfreeze ALL parameters for FT Phase
    for param in model.model.parameters():
        param.requires_grad = True
        
    # Get model info BEFORE training starts
    model_info_ft = get_yolo_params_info(model)
    print(f"🔥 FT Params: {model_info_ft['trainable_params']:,} trainable (Full Model)")

    results_ft = model.train(
        data=yaml_path,
        epochs=epochs_ft,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr_ft,
        optimizer=optimizer_str,
        weight_decay=weight_decay,
        warmup_epochs=0, 
        patience=early_stop_patience,
        cos_lr=True,
        dropout=dropout_rate,
        mosaic=mosaic_prob,
        freeze=0, 
        project=str(final_out),
        name="phase_ft",
        device=device,
        exist_ok=True,
        save=True
    )

    # --- RESULTS CONSOLIDATION (Unified results.csv) ---
    csv_tl = Path(results_tl.save_dir) / 'results.csv'
    csv_ft = Path(results_ft.save_dir) / 'results.csv'
    
    history = {
        'train_box_loss': [],
        'train_class_loss': [],
        'train_dfl_loss': [],
        'train_loss': [],

        'val_box_loss': [],
        'val_class_loss': [],
        'val_dfl_loss': [],
        'val_loss': [],

        'val_precision': [],
        'val_recall': [],
        'val_f1': [],

        'val_map50': [],
        'val_map': [],

        'lr': []
    }

    if csv_tl.exists() and csv_ft.exists():
        df_tl = pd.read_csv(csv_tl).apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df_ft = pd.read_csv(csv_ft).apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        df_tl.columns = [c.strip() for c in df_tl.columns]
        df_ft.columns = [c.strip() for c in df_ft.columns]
        
        df_master = pd.concat([df_tl, df_ft], ignore_index=True)

        # Explicit global epoch independent from Ultralytics internal epoch indexing
        df_master['global_epoch'] = np.arange(1, len(df_master) + 1)
        df_master['phase'] = ['TL' if i < epochs_tl else 'FT' for i in range(len(df_master))]

        df_master.to_csv(final_out / 'results_complete.csv', index=False)

        history['train_box_loss'] = df_master['train/box_loss'].tolist()
        history['train_class_loss'] = df_master['train/cls_loss'].tolist()
        history['train_dfl_loss'] = df_master['train/dfl_loss'].tolist()

        history['val_box_loss'] = df_master['val/box_loss'].tolist()
        history['val_class_loss'] = df_master['val/cls_loss'].tolist()
        history['val_dfl_loss'] = df_master['val/dfl_loss'].tolist()

        history['train_loss'] = (
            df_master['train/box_loss'] +
            df_master['train/cls_loss'] +
            df_master['train/dfl_loss']
        ).tolist()

        history['val_loss'] = (
            df_master['val/box_loss'] +
            df_master['val/cls_loss'] +
            df_master['val/dfl_loss']
        ).tolist()

        history['val_precision'] = df_master['metrics/precision(B)'].tolist()
        history['val_recall'] = df_master['metrics/recall(B)'].tolist()

        precision = df_master['metrics/precision(B)'].astype(float)
        recall = df_master['metrics/recall(B)'].astype(float)

        history['val_f1'] = (
            2 * precision * recall / (precision + recall + 1e-8)
        ).tolist()

        history['val_map50'] = df_master['metrics/mAP50(B)'].tolist()
        history['val_map'] = df_master['metrics/mAP50-95(B)'].tolist()

        history['lr'] = df_master['lr/pg0'].tolist()
    else:
        print("⚠️ Warning: Could not find TL/FT results.csv files. Skipping consolidated history.")

    # --- FINAL TEST EVALUATION (Explicit) ---
    print("\n✅ Final Evaluation on Test Set...")
    best_final_path = Path(final_out / 'phase_ft') / 'weights' / 'best.pt'
    model = YOLO(best_final_path)

    test_stats = model.val(
        data=yaml_path,
        split='test',
        device=device,
        verbose=False,
        project=str(final_out),
        name="ultralytics_test",
        plots=True,
        exist_ok=True
    )

    # --- COMMON DIAGNOSTICS FOR YOLO ---
    # This creates a confusion matrix using the same logic as EfficientDet:
    # rows = ground truth, columns = predicted, background included.
    yolo_records = collect_yolo_detection_records(
        model=model,
        yaml_path=yaml_path,
        split='test',
        img_size=img_size,
        device=device,
    )

    test_diagnostics = evaluate_detection_diagnostics_from_records(
        records=yolo_records,
        class_names=class_names,
        output_dir=final_out / "test_diagnostics",
        iou_thresh=0.50,
        score_thresh=0.50,
    )

    # --- VISUAL PREDICTIONS ON TEST SET ---
    test_source = read_yaml(yaml_path).get('test')
    test_image_paths = resolve_detection_source(test_source, Path(yaml_path).parent)

    if len(test_image_paths) > 0:
        # We use model.predict() with the specific Inference Arguments from the docs
        model.predict(
            source=test_image_paths,
            conf=0.5,                 # Confidence threshold
            iou=0.45,                 # NMS IoU threshold (default is 0.7, 0.45 is standard)
            imgsz=img_size,           # Inference image size
            save=True,                # Save images with results
            save_txt=False,           # We only want visuals, not labels in txt
            project=str(final_out),   # Output project directory
            name="predicts",          # Sub-folder for this run
            exist_ok=True,            # Do not increment name if folder exists
            device=device,            # Use the same device as training
            line_width=2,             # Bounding box thickness (from Docs)
            show_labels=True,         # Show class names
            show_conf=True            # Show confidence scores
        )
        print(f"✅ Visual results saved in: {final_out / 'predicts'}")
    else:
        print("⚠️ Warning: No 'test' images found in YAML. Skipping visual predictions.")
        
    # --- INFERENCE PROFILING ---
    inference_stats = profile_yolo_inference(model.model, img_size, device_obj)

    test_precision = float(test_stats.results_dict.get('metrics/precision(B)', 0))
    test_recall = float(test_stats.results_dict.get('metrics/recall(B)', 0))
    test_f1 = float(2 * test_precision * test_recall / (test_precision + test_recall + 1e-8))

    # --- SUMMARY GENERATION ---
    summary_data = {
        "metadata": {
            "duration": str(datetime.datetime.now() - start_time).split('.')[0],
            "device": str(device)
        },
        "model_name": architecture,
        "model_size_mb": round(os.path.getsize(best_final_path) / (1024**2), 2),
        "model_info_tl": model_info_tl, 
        "model_info_ft": model_info_ft,
        "epochs_tl": epochs_tl,
        "epochs_ft": epochs_ft,
        "history": history,
        "results_test": {
            "map_50": float(test_stats.results_dict.get('metrics/mAP50(B)', 0)),
            "map": float(test_stats.results_dict.get('metrics/mAP50-95(B)', 0)),
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1
        },
        "test_diagnostics": test_diagnostics,
        "best_val_map50_TL": float(results_tl.results_dict.get('metrics/mAP50(B)', 0)),
        "best_val_map50_FT": float(results_ft.results_dict.get('metrics/mAP50(B)', 0)),
        "test_mAP50": float(test_stats.results_dict.get('metrics/mAP50(B)', 0)),
        "inference_stats": inference_stats
    }

    # Save summary files
    with open(final_out / "summary_training.json", 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
    
    with open(final_out / "summary_training.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(summary_data, f)

    shutil.copy(best_final_path, final_out / 'best.pt')

    # --- TRAINING DASHBOARD WITH CONFUSION MATRIX ---
    plot_yolo_curves(
        csv_path=final_out / 'results_complete.csv',
        output_dir=final_out,
        epochs_tl=epochs_tl,
        confusion_matrix=np.array(test_diagnostics["confusion_matrix_counts"]),
        confusion_labels=test_diagnostics["confusion_matrix_labels"]
    )

    # --- CLEANUP ---
    print("\n🧹 Cleaning memory...")
    del model, results_tl, results_ft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"🎉 Final deliverable generated in {final_out}")
    return summary_data