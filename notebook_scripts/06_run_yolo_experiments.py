import sys
from pathlib import Path
import torch
import os

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from engine.train_yolo import train_yolo_pipeline
from visualization.visualize import plot_yolo_curves
from utils.file_utils import read_yaml

# Config
config = read_yaml(PROJECT_ROOT / 'paths.yaml')
experiments_path = PROJECT_ROOT / config['dataset']['processed_path'] / 'experiments'
list_experiments = sorted([f for f in os.listdir(experiments_path) if os.path.isdir(experiments_path / f)])
selected_models_yolo = config['selected_models_yolo']

# Hyperparameters (igual que en EfficientDet)
EPOCHS_TL, EPOCHS_FT = 10, 20
BATCH_SIZE = 16
OPTIMIZER = 'AdamW'
LR_TL, LR_FT = 1e-3, 1e-4

for id_model, (MODEL_NAME, model_cfg) in enumerate(selected_models_yolo.items()):
    IMG_SIZE = model_cfg['img_size']

    for experiment in list_experiments:
        exp_path = experiments_path / experiment
        fold_configs = sorted(list(exp_path.glob('*config.yaml')))

        for fold_idx, yaml_file in enumerate(fold_configs):
            print(f"\n--- Model: {MODEL_NAME} | Fold: {fold_idx} ---")
            
            # --- 1. RUN NORMAL FOLD ---
            out_dir = PROJECT_ROOT / 'runs' / MODEL_NAME / experiment / yaml_file.stem
            train_yolo_pipeline(
                architecture=MODEL_NAME, yaml_path=str(yaml_file), output_dir=str(out_dir),
                epochs_tl=EPOCHS_TL, epochs_ft=EPOCHS_FT, batch_size=BATCH_SIZE,
                img_size=IMG_SIZE, lr_tl=LR_TL, lr_ft=LR_FT, optimizer_str=OPTIMIZER,
                mosaic_prob=1.0 # Enable augmentations
            )
            plot_yolo_curves(out_dir / 'results.csv', out_dir, EPOCHS_TL)
            
            torch.cuda.empty_cache()