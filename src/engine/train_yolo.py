import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import json
import time
import os
import shutil

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
    mosaic_prob: float = 1.0, 
    device: str = 'cuda'
):
    """
    Advanced YOLO pipeline with full manual control over training phases.
    
    Args:
        architecture (str): Path to weights or model name.
        yaml_path (str): Data configuration.
        output_dir (str): Folder to save this specific fold's results.
        epochs_tl/ft (int): Epochs for frozen/unfrozen phases.
        batch_size (int): Custom batch size.
        optimizer_str (str): Choice of 'Adam', 'AdamW', 'SGD', etc.
    """
    final_out = Path(output_dir)
    final_out.mkdir(parents=True, exist_ok=True)

    # --- PHASE 1: TRANSFER LEARNING (Backbone Frozen) ---
    print(f"\n[YOLO] Starting Phase 1 (TL) | Optimizer: {optimizer_str} | Batch: {batch_size}")
    model = YOLO(architecture)
    
    results_tl = model.train(
        data=yaml_path,
        epochs=epochs_tl,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr_tl,
        optimizer=optimizer_str,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        cos_lr=False, # Active Cosine LR to match EfficientDet
        dropout=dropout_rate,
        mosaic=mosaic_prob,
        freeze=10, 
        project=str(final_out),
        name="phase_tl",
        device=device,
        exist_ok=True,
        save=True,
        plots=True
    )

    # --- PHASE 2: FINE TUNING (Everything Unfrozen) ---
    print(f"\n[YOLO] Starting Phase 2 (FT) | LR: {lr_ft}")
    best_tl_weights = Path(results_tl.save_dir) / 'weights' / 'best.pt'
    model = YOLO(best_tl_weights) 

    results_ft = model.train(
        data=yaml_path,
        epochs=epochs_ft,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr_ft,
        optimizer=optimizer_str,
        weight_decay=weight_decay,
        warmup_epochs=0, 
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
    
    if csv_tl.exists() and csv_ft.exists():
        df_tl = pd.read_csv(csv_tl).apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df_ft = pd.read_csv(csv_ft).apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df_tl.columns = [c.strip() for c in df_tl.columns]
        df_ft.columns = [c.strip() for c in df_ft.columns]
        
        df_ft['epoch'] += epochs_tl
        df_master = pd.concat([df_tl, df_ft], ignore_index=True)
        df_master.to_csv(final_out / 'results.csv', index=False)

    # Move best weights to root for easy access
    best_final = Path(results_ft.save_dir) / 'weights' / 'best.pt'
    if best_final.exists():
        shutil.copy(best_final, final_out / 'best.pt')

    # --- INFERENCE PROFILING (Warning Fixed) ---
    model.to(device); model.eval()
    # Usamos torch.rand (0 a 1) en lugar de randn para evitar el warning de normalización
    dummy = torch.rand(1, 3, img_size, img_size).to(device)
    
    for _ in range(10): _ = model(dummy, verbose=False) # Warmup
    
    t0 = time.perf_counter()
    for _ in range(100): _ = model(dummy, verbose=False)
    avg_inf = (time.perf_counter() - t0) / 100

    summary = {
        "model_name": architecture,
        "model_size_mb": round(os.path.getsize(final_out / 'best.pt') / (1024**2), 2),
        "fps": round(1/avg_inf, 1),
        "metrics": {
            "mAP50": float(results_ft.results_dict.get('metrics/mAP50(B)', 0)),
            "mAP50-95": float(results_ft.results_dict.get('metrics/mAP50-95(B)', 0))
        }
    }
    
    with open(final_out / "summary_training.json", 'w') as f:
        json.dump(summary, f, indent=4)

    return summary