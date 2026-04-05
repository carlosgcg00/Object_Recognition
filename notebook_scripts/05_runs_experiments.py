import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from tqdm.notebook import tqdm
import os
from dataset.efficientdet_dataset import EfficientDetDataset
from models.efficientdet import EfficientDet
from utils.data_augmentations import get_transforms
from utils.file_utils import read_yaml
from engine.train_pipeline import train_efficientdet_pipeline
from utils.data_augmentations import get_transforms


# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))
CONFIG_PATH = PROJECT_ROOT / 'paths.yaml'
config = read_yaml(CONFIG_PATH)
experiments_path = Path(PROJECT_ROOT / config['dataset']['processed_path'] / 'experiments')

# List experiments
list_experiments = sorted([f for f in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, f))])
selected_models_efficientdet = config['selected_models_efficientdet']


# HYPERPARAMETERS
EPOCHS_TL = 10
EPOCHS_FT = 20
BATCH_SIZE = 16
LR_TL = 1e-3
LR_FT = 1e-4
OPTIMIZER = 'adamw' # ['adamw', 'adam', 'nadam', 'sgd']
SCHEDULER = 'cosine' # ['cosine', 'step', 'reduce_on_plateau]
EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 2e-4
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Select model
for id_model, (SELECTED_MODEL, model_config) in enumerate(list(selected_models_efficientdet.items())):
    IMG_SIZE = model_config['img_size']

    # Get transforms
    train_transforms=get_transforms(
        mode='train',
        target_size=IMG_SIZE, 
        random_crop_prob=0.5,
        random_flip_prob=0.5,
        random_coarse_dropout_prob=0.1,
        random_color_jitter_prob=0.4
    )

    MOSAIC_PROB = 0.4
        
    train_transforms_no_augs=get_transforms(
        mode='train',
        target_size=IMG_SIZE, 
        random_crop_prob=0.0,
        random_flip_prob=0.0,
        random_coarse_dropout_prob=0.0,
        random_color_jitter_prob=0.0
        )
    val_transforms = get_transforms('val', IMG_SIZE)

    for idx_experiment, experiment in enumerate(list_experiments):
        experiment_path = experiments_path / experiment
        
        # Buscar archivos yaml
        fold_config_yaml_files = sorted([f for f in experiment_path.glob('*.yaml') if 'config' in f.name])
        
        for fold_index, fold_config_yaml_file in enumerate(fold_config_yaml_files):

            print(f"\n{'='*80}")
            print(f"Model: {SELECTED_MODEL}")
            print(f"Experiment: {experiment}")
            print(f"Fold: {fold_index + 1}/{len(fold_config_yaml_files)}")
            print(f"{'='*80}\n")
            fold_name = fold_config_yaml_file.stem
            OUTPUT_DIR = PROJECT_ROOT / 'runs' / SELECTED_MODEL / experiment / fold_name
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            
            train_efficientdet_pipeline(
                architecture=SELECTED_MODEL,
                img_size=IMG_SIZE,
                epochs_tl=EPOCHS_TL,
                epochs_ft=EPOCHS_FT,
                batch_size=BATCH_SIZE,
                lr_tl=LR_TL,
                lr_ft=LR_FT,
                optimizer_str=OPTIMIZER,
                scheduler_str=SCHEDULER,
                early_stop_patience=EARLY_STOP_PATIENCE,
                warmup_epochs=WARMUP_EPOCHS,
                weight_decay=WEIGHT_DECAY,
                dropout_rate=DROPOUT,
                train_transforms=train_transforms,
                val_transforms=val_transforms,
                mosaic_prob=MOSAIC_PROB,
                output_dir=OUTPUT_DIR,
                yaml_path=fold_config_yaml_file,
                device=DEVICE
            )

            if fold_index == 0 and SELECTED_MODEL == 'efficientdet_d2' and idx_experiment == 0:
                fold_name = fold_config_yaml_file.stem
                OUTPUT_DIR = PROJECT_ROOT / 'runs' / SELECTED_MODEL / experiment / f'no_augs_{fold_name}'
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                
                
                train_efficientdet_pipeline(
                    architecture=SELECTED_MODEL,
                    img_size=IMG_SIZE,
                    epochs_tl=EPOCHS_TL,
                    epochs_ft=EPOCHS_FT,
                    batch_size=BATCH_SIZE,
                    lr_tl=LR_TL,
                    lr_ft=LR_FT,
                    optimizer_str=OPTIMIZER,
                    scheduler_str=SCHEDULER,
                    early_stop_patience=EARLY_STOP_PATIENCE,
                    warmup_epochs=WARMUP_EPOCHS,
                    weight_decay=WEIGHT_DECAY,
                    dropout_rate=DROPOUT,
                    train_transforms=train_transforms_no_augs,
                    val_transforms=val_transforms,
                    mosaic_prob=0.0,
                    output_dir=OUTPUT_DIR,
                    yaml_path=fold_config_yaml_file,
                    device=DEVICE
                )
                
                fold_name = fold_config_yaml_file.stem
                OUTPUT_DIR = PROJECT_ROOT / 'runs' / SELECTED_MODEL / experiment / f'augs_{fold_name}_without_mosaic'
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                
                
                train_efficientdet_pipeline(
                    architecture=SELECTED_MODEL,
                    img_size=IMG_SIZE,
                    epochs_tl=EPOCHS_TL,
                    epochs_ft=EPOCHS_FT,
                    batch_size=BATCH_SIZE,
                    lr_tl=LR_TL,
                    lr_ft=LR_FT,
                    optimizer_str=OPTIMIZER,
                    scheduler_str=SCHEDULER,
                    early_stop_patience=EARLY_STOP_PATIENCE,
                    warmup_epochs=WARMUP_EPOCHS,
                    weight_decay=WEIGHT_DECAY,
                    dropout_rate=DROPOUT,
                    train_transforms=train_transforms,
                    val_transforms=val_transforms,
                    mosaic_prob=0.0,
                    output_dir=OUTPUT_DIR,
                    yaml_path=fold_config_yaml_file,
                    device=DEVICE
                )
