# src/engine/train_efficientdet.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import datetime
import yaml
import gc
import time
import json

# Project imports
from dataset.efficientdet_dataset import EfficientDetDataset
from models.efficientdet import EfficientDet
from utils.data_augmentations import get_transforms
from utils.file_utils import read_yaml, build_ordered_class_dict
from visualization.visualize import plot_training_curves, save_evaluation_grid
from utils.seeds import seed_everything, seed_worker
from evaluations.eval_effdet import (
    evaluate_detection_diagnostics,
    collect_detection_records,
    build_detection_confusion_matrix,
    summarize_confusion_matrix
)


# ==========================================
# UTILITIES
# ==========================================
def collate_fn(batch):
    return tuple(zip(*batch))

def effdet_yxyx_to_xyxy(boxes_tensor):
    """Convert from [y1, x1, y2, x2] (Dataset) to [x1, y1, x2, y2] (Metrics).
    
    Args:
        boxes_tensor (torch.Tensor): Tensor of shape (N, 4) with bounding boxes in yxyx format.
    
    Returns:
        torch.Tensor: Tensor of shape (N, 4) with bounding boxes in xyxy format.
    """
    if boxes_tensor.shape[0] == 0:
        return boxes_tensor
    return boxes_tensor[:, [1, 0, 3, 2]]

def set_optimizer_scheduler(model_params, optimizer_str, scheduler_str, epochs, lr, weight_decay, warmup_epochs):
    """
    Set optimizer and scheduler.
    
    Args:
        model_params (torch.nn.Module): Model parameters.
        optimizer_str (str): Optimizer name.
        scheduler_str (str): Scheduler name.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        warmup_epochs (int): Number of warmup epochs.
    
    Returns:
        tuple: Optimizer, scheduler, warmup_scheduler.
    """
    if optimizer_str == 'adamw':
        optimizer = optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_str == 'adam':
        optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_str == 'sgd':
        optimizer = optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_str} not supported")
    
    if scheduler_str == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_str == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_str == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        scheduler = None
        
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs) if warmup_epochs > 0 else None
    
    return optimizer, scheduler, warmup_scheduler

def profile_inference_speed(model, dataloader, device, num_warmup_batches=3):
    """
    Measure inference latency of the model on the GPU.
    Returns statistics per batch and estimated per single image.
    
    Args:
        model (torch.nn.Module): Model to profile.
        dataloader (torch.utils.data.DataLoader): DataLoader to use for profiling.
        device (torch.device): Device to use for profiling.
        num_warmup_batches (int): Number of warmup batches to use for profiling.
    
    Returns:
        dict: Dictionary with inference speed statistics.
    """
    model.eval()
    model.switch_to_predict() # Ensure NMS mode
    
    batch_times = []
    
    # PyTorch CUDA events to measure exact GPU time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    print(f"\n⏱️  Profiling inference latency on {device}...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader, desc="Measuring latency")):
            images_stack = torch.stack(images).to(device).float()
            
            # --- WARMUP ---
            if i < num_warmup_batches:
                _ = model(images_stack)
                continue
            
            # --- REAL MEASUREMENT ---
            if device == 'cuda':
                starter.record()
                _ = model(images_stack)
                ender.record()
                
                torch.cuda.synchronize() # Wait for GPU to finish
                curr_time = starter.elapsed_time(ender) / 1000.0 # Convert from ms to seconds
            else:
                # Fallback for CPU
                t0 = time.perf_counter()
                _ = model(images_stack)
                curr_time = time.perf_counter() - t0
                
            batch_times.append(curr_time)
            
    # Calculate statistics
    batch_times = np.array(batch_times)
    batch_size = dataloader.batch_size
    
    stats = {
        'batch_size': batch_size,
        'mean_batch_time_sec': np.mean(batch_times),
        'std_batch_time_sec': np.std(batch_times),
        'mean_img_time_sec': np.mean(batch_times) / batch_size,
        'fps_batch_based': batch_size / np.mean(batch_times) 
    }
    
    print(f"   ► Mean per BATCH (size={batch_size}): {stats['mean_batch_time_sec']*1000:.2f} ms ± {stats['std_batch_time_sec']*1000:.2f} ms")
    print(f"   ► Estimated time per IMAGE: {stats['mean_img_time_sec']*1000:.2f} ms")
    print(f"   ► Max FPS: {stats['fps_batch_based']:.1f} FPS")
    
    return stats

def validate_precision_recall_f1(
    model,
    dataloader,
    device,
    class_names,
    iou_thresh: float = 0.50,
    score_thresh: float = 0.50
):
    """
    Computes validation precision, recall and F1 at a fixed IoU and confidence threshold.

    This is complementary to mAP:
    - mAP evaluates performance across IoU/score thresholds.
    - This function gives YOLO-like epoch curves for precision/recall/F1 at fixed thresholds.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Validation or test DataLoader.
        device (torch.device): Device to use.
        class_names (dict): Mapping from class IDs to class names.
        iou_thresh (float): IoU threshold used to match predictions with ground truth.
        score_thresh (float): Confidence threshold used to filter predictions.

    Returns:
        tuple: global_metrics, per_class_df
    """
    records = collect_detection_records(
        model=model,
        dataloader=dataloader,
        device=device
    )

    cm = build_detection_confusion_matrix(
        records=records,
        class_names=class_names,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh
    )

    per_class_df, global_metrics = summarize_confusion_matrix(
        cm=cm,
        class_names=class_names
    )

    return global_metrics, per_class_df

def validate_map(model, dataloader, device):
    """Calculates mAP using pure NMS predictions.
    
    Args:
        model (torch.nn.Module): Model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader to use for validation.
        device (torch.device): Device to use for validation.
    
    Returns:
        dict: Dictionary with mAP statistics.
    """
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
    val_pbar = tqdm(dataloader, desc="[3/3] Validating mAP", leave=False)
    
    with torch.no_grad():
        for images, targets in val_pbar:
            images_stack = torch.stack(images).to(device).float()
            detections = model(images_stack)
            
            preds_list, target_list = [], []
            for i in range(len(images)):
                res = detections[i]
                if res is not None and res.shape[0] > 0:
                    preds_list.append({
                        'boxes': res[:, :4],
                        'scores': res[:, 4],
                        'labels': res[:, 5].to(torch.int64)
                    })
                else:
                    preds_list.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'scores': torch.empty((0,), device=device),
                        'labels': torch.empty((0,), device=device, dtype=torch.int64)
                    })

                gt_boxes_xyxy = effdet_yxyx_to_xyxy(targets[i]['bbox'].to(device))
                target_list.append({
                    'boxes': gt_boxes_xyxy,
                    'labels': targets[i]['cls'].to(device).to(torch.int64)
                })

            metric.update(preds_list, target_list)
            
    return metric.compute(), metric

def safe_metric(value, default: float = 0.0) -> float:
    """
    Converts None, NaN or infinite metric values to a safe float.

    Args:
        value: Metric value.
        default (float): Value to return if metric is None, NaN or infinite.

    Returns:
        float: Safe metric value.
    """
    if value is None:
        return default

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if np.isfinite(value) else default

# ==========================================
# PRINCIPAL FUNCTION
# ==========================================
def train_efficientdet_pipeline(
        yaml_path: str,
        output_dir: str,
        epochs_tl: int = 5,
        epochs_ft: int = 30,
        batch_size: int = 8,
        img_size: int = 640,
        lr_tl: float = 1e-3,
        lr_ft: float = 1e-4,
        weight_decay: float = 1e-4,
        architecture: str = 'tf_efficientdet_d1',
        warmup_epochs: int = 0,
        early_stop_patience: int = 20,
        optimizer_str: str = 'adamw',
        scheduler_str: str = 'cosine',
        dropout_rate: float = 0.2,
        mosaic_prob: float = 0.4,
        train_transforms=None,
        val_transforms=None,
        device=None,
        seed: int = 42,
        num_workers: int = 4,
        class_order: list = None
    ):
    """Trains an EfficientDet model using the specified configuration.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        output_dir (str): Directory to save the trained model.
        epochs_tl (int): Number of epochs for transfer learning.
        epochs_ft (int): Number of epochs for fine-tuning.
        batch_size (int): Batch size for training.
        img_size (int): Image size for training.
        lr_tl (float): Learning rate for transfer learning.
        lr_ft (float): Learning rate for fine-tuning.
        weight_decay (float): Weight decay for the optimizer.
        architecture (str): Architecture to use for training.
        warmup_epochs (int): Number of warmup epochs.
        early_stop_patience (int): Number of epochs to wait before early stopping.
        optimizer_str (str): Optimizer to use for training.
        scheduler_str (str): Scheduler to use for training.
        dropout_rate (float): Dropout rate for the model.
        mosaic_prob (float): Probability of using mosaic augmentation.
        train_transforms (callable): Transforms to apply to the training data.
        val_transforms (callable): Transforms to apply to the validation data.
        device (torch.device): Device to use for training.
        seed (int): Seed for reproducibility.
        num_workers (int): Number of worker processes for data loading.
        class_order (list): Order of classes in the dataset.
    
    Returns:
        dict: Dictionary with training results.
    """
    generator = seed_everything(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Data Configuration
    config = read_yaml(yaml_path)
    class_names = build_ordered_class_dict(
        dataset_names=config['names'],
        class_order=class_order if class_order is not None else config['names'],
        one_based=True
    ) 
    num_classes = len(class_names)

    if train_transforms is None:
        train_transforms = get_transforms(mode='train', target_size=img_size)
    if val_transforms is None:
        val_transforms = get_transforms(mode='val', target_size=img_size)
    
    train_ds = EfficientDetDataset(
        Path(config['train']), 
        img_size, 
        train_transforms, 
        is_train=True, 
        mosaic_prob=mosaic_prob)
    val_ds = EfficientDetDataset(
        Path(config['val']), 
        img_size, 
        val_transforms, 
        is_train=False, 
        mosaic_prob=0.0)
    test_ds = EfficientDetDataset(
        Path(config['test']), 
        img_size, 
        val_transforms, 
        is_train=False, 
        mosaic_prob=0.0)
    
    worker_init = seed_worker if num_workers > 0 else None
    persistent_workers = num_workers > 0
    pin_memory = device == 'cuda'
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        persistent_workers=persistent_workers
    )
    
    # 2. Model Initialization
    model = EfficientDet(num_classes=num_classes, img_size=img_size, architecture=architecture, is_training=True, pretrained=True, dropout_rate = dropout_rate).to(device)
    
    # Metric History (Restored for plot_training_curves)
    history = {
        'train_loss': [], 'train_class_loss': [], 'train_box_loss': [],
        'val_loss': [], 'val_class_loss': [], 'val_box_loss': [],

        # Detection metrics over epochs
        'val_map': [],
        'val_map50': [],
        'val_map75': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],

        'lr': [],

        # Per-class mAP. Note: torchmetrics map_per_class is mAP@[.50:.95], not mAP@50.
        'val_map_per_class': {c_id: [] for c_id in class_names.keys()}
    }

    best_map50_TL = 0.0
    best_map50_FT = 0.0
    early_stop_counter_FT = 0
    start_time = datetime.datetime.now()
    total_epochs = epochs_tl + epochs_ft

    model_info_tl = {}
    model_info_ft = {}

    print(f"\n🚀 STARTING TRAINING: {epochs_tl} epochs TL + {epochs_ft} epochs FT en {device}\n")       

    for epoch in range(total_epochs):
        # -- PHASE CONFIGURATION --
        if epoch == 0:
            print("\n" + "="*60 + f"\nPhase 1: Transfer Learning (TL) - {epochs_tl} epochs\n" + "="*60)
            model.freeze_backbone()
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer, scheduler, warmup_scheduler = set_optimizer_scheduler(
                model_params = trainable_params, 
                optimizer_str = optimizer_str, 
                scheduler_str = None, 
                epochs = epochs_tl, 
                lr = lr_tl, 
                weight_decay = weight_decay, 
                warmup_epochs = warmup_epochs
            )

            model_info_tl = model.get_model_info()
            
        elif epoch == epochs_tl:
            print("\n" + "="*60 + f"\nPhase 2: Fine Tuning (FT) - {epochs_ft} epochs\n" + "="*60)
            model.unfreeze_all()
            optimizer, scheduler, warmup_scheduler = set_optimizer_scheduler(
                model_params = model.parameters(), 
                optimizer_str = optimizer_str, 
                scheduler_str = scheduler_str, 
                epochs = epochs_ft, 
                lr = lr_ft, 
                weight_decay = weight_decay, 
                warmup_epochs = 0
            )

            model_info_ft = model.get_model_info()
        
        phase_name = "TL" if epoch < epochs_tl else "FT"
            
        # ==========================================
        # [1/3] TRAINING PHASE
        # ==========================================
        model.switch_to_train()
        model.train() 
        epoch_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [1/3] Train Loss", leave=False)

        for images, targets in train_pbar:
            images_stack = torch.stack(images).to(device).float()
            
            # Force the same scale and size for loss stability
            batch_targets = {
                'bbox': [t['bbox'].to(device) for t in targets],
                'cls': [t['cls'].to(device) for t in targets],
                'img_scale': torch.tensor([1.0]*len(targets)).to(device),
                'img_size': torch.tensor([[img_size, img_size]]*len(targets)).to(device)
            }

            optimizer.zero_grad()
            loss_dict = model(images_stack, batch_targets)
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_losses['total'] += loss_dict['loss'].item()
            epoch_losses['class'] += loss_dict['class_loss'].item()
            epoch_losses['box'] += loss_dict['box_loss'].item()
            
            train_pbar.set_postfix({'loss': f"{loss_dict['loss'].item():.4f}"})
            
        # ==========================================
        # [2/3] VALIDATION PHASE (LOSS)
        # ==========================================
        model.switch_to_train() 
        model.eval()            
        val_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        
        val_loss_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [2/3] Val Loss", leave=False)
        
        with torch.no_grad():
            for images, targets in val_loss_pbar:
                images_stack = torch.stack(images).to(device).float()
                
                # Same strict format
                batch_targets = {
                    'bbox': [t['bbox'].to(device) for t in targets],
                    'cls': [t['cls'].to(device) for t in targets],
                    'img_scale': torch.tensor([1.0]*len(targets)).to(device),
                    'img_size': torch.tensor([[img_size, img_size]]*len(targets)).to(device)
                }
                
                loss_dict = model(images_stack, batch_targets)
                val_losses['total'] += loss_dict['loss'].item()
                val_losses['class'] += loss_dict['class_loss'].item()
                val_losses['box'] += loss_dict['box_loss'].item()

        # ==========================================
        # [3/3] VALIDATION PHASE (mAP)
        # ==========================================
        model.switch_to_predict() 
        model.eval()
                
        val_res, metric_obj = validate_map(model, val_loader, device)
        
        print("Observed classes", val_res["classes"])
        print("mAP per classes", val_res["map_per_class"])

        current_map50 = val_res['map_50'].item()
        current_map75 = val_res['map_75'].item()
        current_map = val_res['map'].item()

        # YOLO-like validation curves at fixed IoU and score threshold
        val_pr_global, val_pr_per_class = validate_precision_recall_f1(
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=class_names,
            iou_thresh=0.50,
            score_thresh=0.50
        )

        current_precision = safe_metric(val_pr_global["precision_micro"])
        current_recall = safe_metric(val_pr_global["recall_micro"])
        current_f1 = safe_metric(val_pr_global["f1_micro"])

        # ==========================================
        # UPDATE HISTORY AND SCHEDULERS
        # ==========================================
        if phase_name == "TL" and warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler:
            scheduler.step()

        # Calculate averages
        avg_train_loss = epoch_losses['total'] / len(train_loader)
        avg_val_loss = val_losses['total'] / len(val_loader)

        # Save History
        history['train_loss'].append(float(avg_train_loss))
        history['train_class_loss'].append(float(epoch_losses['class'] / len(train_loader)))
        history['train_box_loss'].append(float(epoch_losses['box'] / len(train_loader)))
        
        history['val_loss'].append(float(avg_val_loss))
        history['val_class_loss'].append(float(val_losses['class'] / len(val_loader)))
        history['val_box_loss'].append(float(val_losses['box'] / len(val_loader)))
        
        history['val_map50'].append(float(current_map50))
        history['val_map75'].append(float(current_map75))
        history['val_map'].append(float(current_map))

        history['val_precision'].append(float(current_precision))
        history['val_recall'].append(float(current_recall))
        history['val_f1'].append(float(current_f1))

        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        
        map_classes = val_res.get('map_per_class', torch.zeros(num_classes))
        for idx, c_id in enumerate(sorted(class_names.keys())):
            val_cls = map_classes[idx].item() if idx < len(map_classes) else 0.0
            history['val_map_per_class'][c_id].append(float(val_cls))

        print(
            f"📊 Epoch {epoch+1} [{phase_name}]: "
            f"TrainLoss={avg_train_loss:.4f} | "
            f"ValLoss={avg_val_loss:.4f} | "
            f"mAP@50={current_map50:.4f} | "
            f"Precision={current_precision:.4f} | "
            f"Recall={current_recall:.4f} | "
            f"F1={current_f1:.4f}"
        )        
        # ==========================================
        # CHECKPOINTS AND EARLY STOPPING
        # ==========================================
        save_dict = {'model_state_dict': model.state_dict(), 'history': history}
        torch.save(save_dict, out_dir / 'last.pt')
        
        if phase_name == "TL":
            if current_map50 >= best_map50_TL:
                best_map50_TL = current_map50
                torch.save(save_dict, out_dir / 'best_TL.pt')
        else: # Phase FT
            if current_map50 >= best_map50_FT:
                best_map50_FT = current_map50
                early_stop_counter_FT = 0
                torch.save(save_dict, out_dir / 'best_FT.pt')
                print("🏆 New Best mAP@50 (Fine-Tuning)!")
            else:
                early_stop_counter_FT += 1
                if early_stop_counter_FT >= early_stop_patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}"); break

    # ==========================================
    # TEST FINAL AND VISUALIZATION
    # ==========================================
    print("\n✅ Final Evaluation on Test Set...")
    best_model_path = out_dir / 'best_FT.pt' if (out_dir / 'best_FT.pt').exists() else out_dir / 'best_TL.pt'
        
    # Use load_checkpoint from your model to clean prefixes cleanly
    model.load_checkpoint(best_model_path)
    model.switch_to_predict()
    model.eval()
    
    test_out_dir = out_dir / 'test_predictions'
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Quantitative test evaluation: mAP, mAP@50, mAP@75
    test_res, test_metric = validate_map(model, test_loader, device)

    # Additional test diagnostics:
    # - Detection confusion matrix
    # - Precision, recall and F1
    # - Precision-recall curves
    # - Precision/Recall/F1 vs confidence threshold
    test_diagnostics = evaluate_detection_diagnostics(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=out_dir / "test_diagnostics",
        iou_thresh=0.50,
        score_thresh=0.50,
    )
    
    # Generate visualizations of the Test Set
    with torch.no_grad():
        img_counter = 0
        for images, targets in tqdm(test_loader, desc="Saving Test Images"):
            images_stack = torch.stack(images).to(device).float()
            detections = model(images_stack)

            for b_idx in range(len(images)):
                d = detections[b_idx]
                save_evaluation_grid(
                    images[b_idx],
                    targets[b_idx],
                    d,
                    class_names,
                    test_out_dir / f"test_sample_{img_counter}.jpg",
                    score_thresh=0.5
                )
                img_counter += 1

    # Print test results
    print(
        f"\n📈 Test Results: "
        f"mAP@50={test_res['map_50']:.4f} | "
        f"mAP@75={test_res['map_75']:.4f} | "
        f"mAP={test_res['map']:.4f}\n"
    )

    test_precision = safe_metric(
        test_diagnostics["global_metrics"].get("precision_micro"),
        default=0.0
    )

    test_recall = safe_metric(
        test_diagnostics["global_metrics"].get("recall_micro"),
        default=0.0
    )

    test_f1 = safe_metric(
        test_diagnostics["global_metrics"].get("f1_micro"),
        default=0.0
    )

    print(
        f"📊 Test Diagnostics: "
        f"Precision={test_precision:.4f} | "
        f"Recall={test_recall:.4f} | "
        f"F1={test_f1:.4f}"
    )
    
    # ==========================================
    # 4. INFERENCE PROFILING (NEW)
    # ==========================================
    print("\n" + "="*60)
    print("4. INFERENCE PROFILING (NEW)")
    print("="*60)
    
    # Use the test_loader to profile with real data
    inference_stats = profile_inference_speed(model, test_loader, device, num_warmup_batches=5)
    
    # SAVE YAML REPORT
    results_test_clean = {
        k: (v.item() if v.numel() == 1 else v.tolist())
        for k, v in test_res.items()
        if isinstance(v, torch.Tensor)
    }
    
    summary_data = {
        "metadata": {
            "duration": str(datetime.datetime.now() - start_time).split('.')[0],
            "device": str(device)
        },
        "seed": seed,
        "num_workers": num_workers,
        "model_name": architecture,
        "model_info_tl": model_info_tl,
        "model_info_ft": model_info_ft,
        "epochs_tl": epochs_tl,
        "epochs_ft": epochs_ft,
        "history": history,
        "results_test": results_test_clean,
        "test_diagnostics": test_diagnostics,
        "best_val_map50_TL": float(best_map50_TL),
        "best_val_map50_FT": float(best_map50_FT),
        "test_mAP50": float(test_res['map_50']),
        "early_stop_triggered": early_stop_counter_FT >= early_stop_patience,
        "inference_stats": inference_stats
    }
    # Save in a JSON file
    with open(out_dir / "summary_training.json", 'w') as f:
        json.dump(summary_data, f, indent=4)

    with open(out_dir / "summary_training.yaml", 'w') as f:
        yaml.dump(summary_data, f)
    
    try:
        plot_training_curves(
            history=history,
            output_dir=out_dir,
            epochs_TL=epochs_tl,
            class_names=class_names,
            confusion_matrix=np.array(test_diagnostics["confusion_matrix_counts"]),
            confusion_labels=test_diagnostics["confusion_matrix_labels"]
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not generate plots ({e})")
        
    print(f"🎉 Final entregable generated in {out_dir}")
    
    # ==========================================
    # CLEANUP
    # =========================================
    del model, optimizer, scheduler, warmup_scheduler, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
    del test_metric, metric_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("🧹 Memory cleaned, pipeline finished.")
    return summary_data