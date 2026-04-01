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

# Importaciones locales
from dataset.efficientdet_dataset import EfficientDetDataset
from models.efficientdet import EfficientDet
from utils.data_augmentations import get_transforms
from utils.file_utils import read_yaml
from visualization.visualize import plot_training_curves, save_evaluation_grid

def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    images = torch.stack(images)
    return images, targets

def effdet_yxyx_to_xyxy(boxes_tensor):
    """Convierte de [y1, x1, y2, x2] (Dataset) a [x1, y1, x2, y2] (Metrics)."""
    if boxes_tensor.shape[0] == 0:
        return boxes_tensor
    return boxes_tensor[:, [1, 0, 3, 2]]

def set_optimizer_scheduler(model_params, optimizer_str, scheduler_str, epochs, lr, weight_decay, warmup_epochs):
    """Set optimizer and scheduler.
    
    Args:
        model_params: Parameters of the model.
        optimizer_str: Name of the optimizer.
        scheduler_str: Name of the scheduler.
        epochs: Number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        warmup_epochs: Number of warmup epochs.
    
    Returns:
        Tuple of (optimizer, scheduler, warmup_scheduler).
    """
    if optimizer_str == 'adamw':
        optimizer = optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_str == 'nadam':
        optimizer = optim.NAdam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_str == 'adam':
        optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_str == 'sgd':
        optimizer = optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_str} not supported")
    
    if scheduler_str == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_str == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        scheduler = None
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs) if warmup_epochs > 0 else None
    
    return optimizer, scheduler, warmup_scheduler

def train_efficientdet(
    yaml_path: str,
    output_dir: str,
    epochs_tl: int = 10,
    epochs_ft: int = 30,
    batch_size: int = 16,
    img_size: int = 512,
    lr_tl: float = 1e-3,
    lr_ft: float = 1e-4,
    weight_decay: float = 1e-4,
    architecture: str = 'tf_efficientdet_d1',
    warmup_epochs: int = 5,
    early_stop_patience: int = 20,
    optimizer_str: str = 'adamw',
    scheduler_str: str = 'cosine',
    dropout_rate: float = 0.2,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Configuración de datos
    config = read_yaml(yaml_path)
    class_names = {i+1: name for i, name in enumerate(config['names'])}
    num_classes = len(class_names)
    
    train_ds = EfficientDetDataset(Path(config['train']), img_size, get_transforms('train', img_size))
    val_ds = EfficientDetDataset(Path(config['val']), img_size, get_transforms('val', img_size))
    test_ds = EfficientDetDataset(Path(config['test']), img_size, get_transforms('val', img_size))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 2. Modelo e Infraestructura
    model = EfficientDet(num_classes=num_classes, 
                         img_size=img_size, 
                         architecture=architecture, 
                         is_training=True, 
                         dropout_rate=dropout_rate
                         ).to(device)
    
    # History and Metrics
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
    history = {
        'train_loss': [], 'train_class_loss': [], 'train_box_loss': [],
        'val_loss': [], 'val_class_loss': [], 'val_box_loss': [],
        'val_map': [], 'val_map50': [], 'val_map75': [], 'lr': [],
        'val_map50_per_class': {c_id: [] for c_id in class_names.keys()}
    }

    best_map50_TL = 0.0
    best_map50_FT = 0.0
    early_stop_counter = 0
    start_time = datetime.datetime.now()

    total_epochs = epochs_tl + epochs_ft

    # Initializing optimizer and scheduler
    optimizer, scheduler, warmup_scheduler = None, None, None
    
    # ==================================================
    # ---- LOOP DE ENTRENAMIENTO ----
    # ==================================================
    print(f"\n🚀 STARTING TRAINING: {epochs_tl} epochs TL + {epochs_ft} epochs FT\n")       

    for epoch in range(total_epochs):
        total_params = sum(p.numel() for p in model.parameters())
        # If is the initial epoch
        if epoch == 0:
            print("\n" + "="*60)
            print(f"Phase 1: Transfer Learning (TL) - {epochs_tl} epochs")
            print("="*60)
            model.freeze_backbone()
            trainable_params = model.get_trainable_params()
            print(f"📊 Parámetros Totales: {total_params:,}")
            print(f"🎯 Parámetros Entrenables (Solo Heads): {trainable_params:,}")
            print(f"🔒 Porcentaje congelado: {100 * (1 - trainable_params/total_params):.2f}%")

            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer, scheduler, warmup_scheduler = set_optimizer_scheduler(
                model_params=trainable_params,
                optimizer_str=optimizer_str,
                scheduler_str=None,
                epochs=epochs_tl,
                lr=lr_tl,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs
            )
        elif epoch == epochs_tl:
            print("\n" + "="*60)
            print(f"Phase 2: Fine Tuning (FT) - {epochs_ft} epochs")
            print("="*60)
            model.unfreeze_all()
            trainable_params = model.get_trainable_params()
            print(f"📊 Parámetros Totales: {total_params:,}")
            print(f"🎯 Parámetros Entrenables (Solo Heads): {trainable_params:,}")
            print(f"🔒 Porcentaje congelado: {100 * (1 - trainable_params/total_params):.2f}%")

            trainable_params = model.parameters()
            optimizer, scheduler, warmup_scheduler = set_optimizer_scheduler(
                model_params=trainable_params,
                optimizer_str=optimizer_str,
                scheduler_str=scheduler_str,
                epochs=epochs_ft,
                lr=lr_ft,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs
            )
        
        phase_name = "TL" if epoch < epochs_tl else "FT"
            

        # -- Fase: Train --
        model.switch_to_train()
        model.train()
        epoch_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase_name}]", leave=False)

        for images, targets in train_pbar:
            images = images.to(device)
            target_dict = {
                "bbox": [t["bbox"].to(device) for t in targets],
                "cls": [t["cls"].to(device) for t in targets],
                "img_size": torch.stack([t["img_size"] for t in targets]).to(device),
                "img_scale": torch.stack([t["img_scale"] for t in targets]).to(device),
            }

            optimizer.zero_grad()
            loss_dict = model(images, target_dict)
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_losses['total'] += loss_dict['loss'].item()
            epoch_losses['class'] += loss_dict['class_loss'].item()
            epoch_losses['box'] += loss_dict['box_loss'].item()
            
            train_pbar.set_postfix({
                'total_loss': f"{loss_dict['loss'].item():.4f}",
                'class_loss': f"{loss_dict['class_loss'].item():.4f}",
                'box_loss': f"{loss_dict['box_loss'].item():.4f}"
            })
            

        # -- Fase: Validation --
        model.switch_to_train()
        model.eval()
        val_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                target_dict = {
                    "bbox": [t["bbox"].to(device) for t in targets],
                    "cls": [t["cls"].to(device) for t in targets],
                    "img_size": torch.stack([t["img_size"] for t in targets]).to(device),
                    "img_scale": torch.stack([t["img_scale"] for t in targets]).to(device),
                }
                
                loss_dict = model(images, target_dict)
                val_losses['total'] += loss_dict['loss'].item()
                val_losses['class'] += loss_dict['class_loss'].item()
                val_losses['box'] += loss_dict['box_loss'].item()
                
        # -- Fase: Val MAP --
        model.switch_to_predict()
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val MAP]", leave=False)
            for images, targets in val_bar:
                images = images.to(device)
                detections = model(images)
                preds, gts = [], []
                for b_idx in range(len(images)):
                    d = detections[b_idx]
                    if d is not None and d.shape[0] > 0:
                        preds.append({"boxes": d[:, :4], "scores": d[:, 4], "labels": d[:, 5].to(torch.int64)})
                    else:
                        preds.append({"boxes": torch.empty((0, 4), device=device), "scores": torch.empty((0,), device=device), "labels": torch.empty((0,), device=device, dtype=torch.int64)})
                    
                    gts.append({"boxes": effdet_yxyx_to_xyxy(targets[b_idx]['bbox']).to(device), "labels": targets[b_idx]["cls"].to(device).to(torch.int64)})
                
                metric.update(preds, gts)
        
        val_res = metric.compute()
        current_map50 = val_res['map_50'].item()
        current_map75 = val_res['map_75'].item()
        current_map = val_res['map'].item()

        
        # -- Aplicación Schedulers --
        # If we are in TL and its warm up, next step in warmup
        if phase_name == "TL" and warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        # If it is not warmup, next step in the scheduler
        else:
            if scheduler:
                if scheduler_str == 'plateau':
                    scheduler.step(epoch_losses['total'])
                else:
                    scheduler.step()

        # -- HISTORY
        history['train_loss'].append(epoch_losses['total'] / len(train_loader))
        history['train_class_loss'].append(epoch_losses['class'] / len(train_loader))
        history['train_box_loss'].append(epoch_losses['box'] / len(train_loader))
        history['val_loss'].append(val_losses['total'] / len(val_loader))
        history['val_class_loss'].append(val_losses['class'] / len(val_loader))
        history['val_box_loss'].append(val_losses['box'] / len(val_loader))
        history['val_map50'].append(current_map50)
        history['val_map75'].append(current_map75)
        history['val_map'].append(current_map)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        map_classes = val_res.get('map_per_class', torch.zeros(num_classes))
        for idx, c_id in enumerate(sorted(class_names.keys())):
            val_cls = map_classes[idx].item() if idx < len(map_classes) else 0.0
            history['val_map50_per_class'][c_id].append(val_cls)


        print(f"📊 Epoch {epoch+1} [{phase_name}]: TrainLoss={history['train_loss'][-1]:.4f} | ValLoss={history['val_loss'][-1]:.4f} | mAP50={current_map50:.4f}")
        
        
        # --- CHECKPOINTS Y EARLY STOPPING ---
        save_dict = {'model_state_dict': model.state_dict(), 'history': history}
        torch.save(save_dict, out_dir / 'last.pt')
        
        if phase_name == "TL":
            if current_map50 > best_map50_TL:
                best_map50_TL = current_map50
                torch.save(save_dict, out_dir / 'best_TL.pt')
        else: # Phase FT
            if current_map50 > best_map50_FT:
                best_map50_FT = current_map50
                early_stop_counter_FT = 0
                torch.save(save_dict, out_dir / 'best_FT.pt')
                print("🏆 New Best mAP@50 (Fine-Tuning)!")
            else:
                early_stop_counter_FT += 1
                if early_stop_counter_FT >= early_stop_patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}"); break

    # ===================================
    # ---- TEST FINAL AND VISUALIZATION ----
    # ===================================
    print("\n✅ Final Evaluation...")
    best_model_path = out_dir / 'best_FT.pt'
    if not best_model_path.exists():
        best_model_path = out_dir / 'best_TL.pt'
        
    model.load_checkpoint(best_model_path)
    model.switch_to_predict()
    metric.reset()
    
    test_out_dir = out_dir / 'test_predictions'
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        img_counter = 0
        for images, targets in tqdm(test_loader, desc="TEST SET"):
            images = images.to(device)
            detections = model(images)
            
            preds, gts = [], []
            for b_idx in range(len(images)):
                d = detections[b_idx]
                if d is not None and d.shape[0] > 0:
                    preds.append({"boxes": d[:, :4], "scores": d[:, 4], "labels": d[:, 5].to(torch.int64)})
                else:
                    preds.append({"boxes": torch.empty((0, 4), device=device), "scores": torch.empty((0,), device=device), "labels": torch.empty((0,), device=device, dtype=torch.int64)})
                
                gts.append({"boxes": effdet_yxyx_to_xyxy(targets[b_idx]['bbox']).to(device), "labels": targets[b_idx]["cls"].to(device).to(torch.int64)})
                
                # SAVE every batch predictions
                save_evaluation_grid(images[b_idx], targets[b_idx], d, class_names, test_out_dir / f"test_sample_{img_counter}.jpg", score_thresh=0.25)
                img_counter += 1
                
            metric.update(preds, gts)

    # SAVE YAML REPORT
    test_res = metric.compute()
    results_test_clean = {k: (v.item() if v.numel()==1 else v.tolist()) for k, v in test_res.items() if isinstance(v, torch.Tensor)}
    
    summary_data = {
        "metadata": {"duration": str(datetime.datetime.now() - start_time).split('.')[0], "device": str(device)},
        "model_name": architecture,
        "optimizer": optimizer_str,
        "scheduler": scheduler_str,
        "epochs_tl": epochs_tl,
        "epochs_ft": epochs_ft,
        "history": history,
        "results_test": results_test_clean,
        "best_val_map50_TL": float(best_map50_TL),
        "best_val_map50_FT": float(best_map50_FT)
    }
    with open(out_dir / "summary_training.yaml", 'w') as f:
        yaml.dump(summary_data, f)
    
    plot_training_curves(history, out_dir, epochs_tl, class_names)
        
    print(f"🎉 Entregable final generado en {out_dir}")
        




    # # 3. Historial y Métricas
    # metric = MeanAveragePrecision(box_format='xyxy').to(device)
    # history = {
    #     'train_loss': [], 'train_class_loss': [], 'train_box_loss': [],
    #     'val_loss': [], 'val_class_loss': [], 'val_box_loss': [],
    #     'val_map50': [], 'val_map75': [], 'lr': []
    # }

    # best_map50 = 0.0
    # early_stop_counter = 0
    # start_time = datetime.datetime.now()

    # # ==================================================
    # # ---- LOOP DE ENTRENAMIENTO ----
    # # ==================================================
    # for epoch in range(epochs):
    #     # --- FASE: TRAIN ---
    #     model.switch_to_train()
    #     model.train()
    #     epoch_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        
    #     train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", leave=False)
    #     for images, targets in train_pbar:
    #         images = images.to(device)
    #         target_dict = {
    #             "bbox": [t["bbox"].to(device) for t in targets],
    #             "cls": [t["cls"].to(device) for t in targets],
    #             "img_size": torch.stack([t["img_size"] for t in targets]).to(device),
    #             "img_scale": torch.stack([t["img_scale"] for t in targets]).to(device),
    #         }

    #         optimizer.zero_grad()
    #         loss_dict = model(images, target_dict)
    #         loss_dict['loss'].backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    #         optimizer.step()
            
    #         epoch_losses['total'] += loss_dict['loss'].item()
    #         epoch_losses['class'] += loss_dict['class_loss'].item()
    #         epoch_losses['box'] += loss_dict['box_loss'].item()
    #         train_pbar.set_postfix({'loss': f"{loss_dict['loss'].item():.3f}"})
            
    #     # --- FASE: VAL LOSS ---
    #     model.switch_to_train() # Seguimos en modo Loss para validar pérdida
    #     model.eval()
    #     val_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
    #     with torch.no_grad():
    #         for images, targets in val_loader:
    #             images = images.to(device)
    #             target_dict = {
    #                 "bbox": [t["bbox"].to(device) for t in targets],
    #                 "cls": [t["cls"].to(device) for t in targets],
    #                 "img_size": torch.stack([t["img_size"] for t in targets]).to(device),
    #                 "img_scale": torch.stack([t["img_scale"] for t in targets]).to(device),
    #             }
    #             l_dict = model(images, target_dict)
    #             val_losses['total'] += l_dict['loss'].item()
    #             val_losses['class'] += l_dict['class_loss'].item()
    #             val_losses['box'] += l_dict['box_loss'].item()

    #     # --- FASE: VAL mAP ---
    #     model.switch_to_predict()
    #     metric.reset()
    #     with torch.no_grad():
    #         for images, targets in tqdm(val_loader, desc="[VAL mAP]", leave=False):
    #             images = images.to(device)
    #             detections = model(images)
    #             preds = [{"boxes": d[:, :4], "scores": d[:, 4], "labels": d[:, 5].to(torch.int64)} for d in detections]
    #             gts = [{"boxes": effdet_yxyx_to_xyxy(t['bbox']).to(device), "labels": t["cls"].to(device).to(torch.int64)} for t in targets]
    #             metric.update(preds, gts)
        
    #     val_res = metric.compute()
        
    #     # Registrar Historia
    #     history['train_loss'].append(epoch_losses['total'] / len(train_loader))
    #     history['train_class_loss'].append(epoch_losses['class'] / len(train_loader))
    #     history['train_box_loss'].append(epoch_losses['box'] / len(train_loader))
    #     history['val_loss'].append(val_losses['total'] / len(val_loader))
    #     history['val_class_loss'].append(val_losses['class'] / len(val_loader))
    #     history['val_box_loss'].append(val_losses['box'] / len(val_loader))
    #     history['val_map50'].append(val_res['map_50'].item())
    #     history['val_map75'].append(val_res['map_75'].item())
    #     history['lr'].append(optimizer.param_groups[0]['lr'])

    #     print(f"📊 Epoch {epoch+1}: TrainLoss={history['train_loss'][-1]:.4f} | ValLoss={history['val_loss'][-1]:.4f} | mAP50={history['val_map50'][-1]:.4f}")

    #     # Schedulers
    #     if warmup_scheduler and epoch < warmup_epochs:
    #         warmup_scheduler.step()
    #     elif scheduler_str == 'plateau':
    #         scheduler.step(val_res['map_50'])
    #     elif scheduler:
    #         scheduler.step()

    #     # Checkpoints
    #     save_dict = {'model_state_dict': model.state_dict(), 'history': history}
    #     torch.save(save_dict, out_dir / 'last.pt')
    #     if history['val_map50'][-1] > best_map50:
    #         best_map50 = history['val_map50'][-1]
    #         early_stop_counter = 0
    #         torch.save(save_dict, out_dir / 'best.pt')
    #         print("🏆 New Best mAP@50!")
    #     else:
    #         early_stop_counter += 1
    #         if early_stop_counter >= early_stop_patience:
    #             print(f"🛑 Early stopping at epoch {epoch+1}"); break

    # # ===================================
    # # ---- TEST FINAL Y VISUALIZACIÓN ----
    # # ===================================
    # print("\n✅ Final Evaluation...")
    # model.load_checkpoint(out_dir / 'best.pt')
    # model.switch_to_predict()
    # metric.reset()
    # test_out_dir = out_dir / 'test_predictions'
    
    # with torch.no_grad():
    #     for i, (images, targets) in enumerate(tqdm(test_loader, desc="TEST SET")):
    #         images = images.to(device)
    #         detections = model(images)
    #         preds = [{"boxes": d[:, :4], "scores": d[:, 4], "labels": d[:, 5].to(torch.int64)} for d in detections]
    #         gts = [{"boxes": effdet_yxyx_to_xyxy(t['bbox']).to(device), "labels": t["cls"].to(device).to(torch.int64)} for t in targets]
    #         metric.update(preds, gts)
            
    #         if i < 10: # Guardar 10 cuadrículas de ejemplo
    #             save_evaluation_grid(images[0], targets[0], detections[0], class_names, test_out_dir / f"test_sample_{i}.jpg")

    # # Guardar Informe YAML
    # test_res = metric.compute()
    # results_test_clean = {k: (v.item() if v.numel()==1 else v.tolist()) for k, v in test_res.items() if isinstance(v, torch.Tensor)}
    
    # summary_data = {
    #     "metadata": {"duration": str(datetime.datetime.now() - start_time).split('.')[0], "device": str(device)},
    #     "results_test": results_test_clean,
    #     "best_val_map50": float(best_map50)
    # }
    # with open(out_dir / "summary_training.yaml", 'w') as f:
    #     yaml.dump(summary_data, f)
    
    # plot_training_curves(history, out_dir)
    # print(f"🎉 Entregable generado en {out_dir}")