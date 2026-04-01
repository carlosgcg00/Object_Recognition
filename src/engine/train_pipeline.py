# src/engine/train_pipeline.py
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

# Importaciones de TU proyecto
from dataset.efficientdet_dataset import EfficientDetDataset
from models.efficientdet import EfficientDet
from utils.data_augmentations import get_transforms
from utils.file_utils import read_yaml
from visualization.visualize import plot_training_curves, save_evaluation_grid

# ==========================================
# UTILIDADES
# ==========================================
def collate_fn(batch):
    return tuple(zip(*batch))

def effdet_yxyx_to_xyxy(boxes_tensor):
    """Convierte de [y1, x1, y2, x2] (Dataset) a [x1, y1, x2, y2] (Metrics)."""
    if boxes_tensor.shape[0] == 0:
        return boxes_tensor
    return boxes_tensor[:, [1, 0, 3, 2]]

def set_optimizer_scheduler(model_params, optimizer_str, scheduler_str, epochs, lr, weight_decay, warmup_epochs):
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
    else:
        scheduler = None
        
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs) if warmup_epochs > 0 else None
    
    return optimizer, scheduler, warmup_scheduler

def validate_map(model, dataloader, device):
    """Calcula mAP usando predicciones NMS puras."""
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

# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def train_efficientdet_pipeline(
    yaml_path: str,
    output_dir: str,
    epochs_tl: int = 5,
    epochs_ft: int = 30,
    batch_size: int = 8,
    img_size: int = 768,
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
    
    # 2. Modelo
    model = EfficientDet(num_classes=num_classes, img_size=img_size, architecture=architecture, is_training=True, pretrained=True, dropout_rate = dropout_rate).to(device)
    
    # Historial de métricas (Restaurado para el plot_training_curves)
    history = {
        'train_loss': [], 'train_class_loss': [], 'train_box_loss': [],
        'val_loss': [], 'val_class_loss': [], 'val_box_loss': [],
        'val_map': [], 'val_map50': [], 'val_map75': [], 'lr': [],
        'val_map50_per_class': {c_id: [] for c_id in class_names.keys()}
    }

    best_map50_TL = 0.0
    best_map50_FT = 0.0
    early_stop_counter_FT = 0
    start_time = datetime.datetime.now()
    total_epochs = epochs_tl + epochs_ft

    print(f"\n🚀 STARTING TRAINING: {epochs_tl} epochs TL + {epochs_ft} epochs FT en {device}\n")       

    for epoch in range(total_epochs):
        # -- CONFIGURACIÓN DE FASES --
        if epoch == 0:
            print("\n" + "="*60 + f"\nPhase 1: Transfer Learning (TL) - {epochs_tl} epochs\n" + "="*60)
            model.freeze_backbone()
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer, scheduler, warmup_scheduler = set_optimizer_scheduler(
                trainable_params, optimizer_str, None, epochs_tl, lr_tl, weight_decay, warmup_epochs
            )
        elif epoch == epochs_tl:
            print("\n" + "="*60 + f"\nPhase 2: Fine Tuning (FT) - {epochs_ft} epochs\n" + "="*60)
            model.unfreeze_all()
            trainable_params = model.parameters()
            optimizer, scheduler, warmup_scheduler = set_optimizer_scheduler(
                trainable_params, optimizer_str, scheduler_str, epochs_ft, lr_ft, weight_decay, warmup_epochs
            )
        
        phase_name = "TL" if epoch < epochs_tl else "FT"
            
        # ==========================================
        # [1/3] FASE DE ENTRENAMIENTO
        # ==========================================
        model.switch_to_train()
        model.train() # Habilita Dropout y BatchNorm
        epoch_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [1/3] Train Loss", leave=False)

        for images, targets in train_pbar:
            images_stack = torch.stack(images).to(device).float()
            
            # Forzamos la misma escala y tamaño para estabilidad del loss
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
        # [2/3] FASE DE VALIDACIÓN (LOSS)
        # ==========================================
        model.switch_to_train() # Usamos DetBenchTrain para que devuelva pérdidas...
        model.eval()            # ...PERO en modo eval para no afectar los pesos/estadísticas!
        val_losses = {'total': 0.0, 'class': 0.0, 'box': 0.0}
        
        val_loss_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [2/3] Val Loss", leave=False)
        
        with torch.no_grad():
            for images, targets in val_loss_pbar:
                images_stack = torch.stack(images).to(device).float()
                
                # Mismo formato estricto
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
        # [3/3] FASE DE VALIDACIÓN (mAP)
        # ==========================================
        model.switch_to_predict() # Cambia a DetBenchPredict (NMS)
        model.eval()
        
        val_res, metric_obj = validate_map(model, val_loader, device)
        current_map50 = val_res['map_50'].item()
        current_map75 = val_res['map_75'].item()
        current_map = val_res['map'].item()

        # ==========================================
        # ACTUALIZACIÓN DE HISTORIAL Y SCHEDULERS
        # ==========================================
        if phase_name == "TL" and warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler:
            scheduler.step()

        # Calcular promedios
        avg_train_loss = epoch_losses['total'] / len(train_loader)
        avg_val_loss = val_losses['total'] / len(val_loader)

        # Guardar Historial
        history['train_loss'].append(avg_train_loss)
        history['train_class_loss'].append(epoch_losses['class'] / len(train_loader))
        history['train_box_loss'].append(epoch_losses['box'] / len(train_loader))
        
        history['val_loss'].append(avg_val_loss)
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

        print(f"📊 Epoch {epoch+1} [{phase_name}]: TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | mAP@50={current_map50:.4f}")
        
        # ==========================================
        # CHECKPOINTS Y EARLY STOPPING
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
        
    # Usamos load_checkpoint de tu modelo para limpiar prefijos limpiamente
    model.load_checkpoint(best_model_path)
    model.switch_to_predict()
    model.eval()
    
    test_out_dir = out_dir / 'test_predictions'
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    test_res, test_metric = validate_map(model, test_loader, device)
    
    # Generar visualizaciones del Test Set
    with torch.no_grad():
        img_counter = 0
        for images, targets in tqdm(test_loader, desc="Saving Test Images"):
            images_stack = torch.stack(images).to(device).float()
            detections = model(images_stack)
            for b_idx in range(len(images)):
                d = detections[b_idx]
                save_evaluation_grid(
                    images[b_idx], targets[b_idx], d, class_names, 
                    test_out_dir / f"test_sample_{img_counter}.jpg", score_thresh=0.25
                )
                img_counter += 1

    # SAVE YAML REPORT
    results_test_clean = {k: (v.item() if v.numel()==1 else v.tolist()) for k, v in test_res.items() if isinstance(v, torch.Tensor)}
    
    summary_data = {
        "metadata": {"duration": str(datetime.datetime.now() - start_time).split('.')[0], "device": str(device)},
        "model_name": architecture,
        "epochs_tl": epochs_tl,
        "epochs_ft": epochs_ft,
        "history": history,
        "results_test": results_test_clean,
        "best_val_map50_TL": float(best_map50_TL),
        "best_val_map50_FT": float(best_map50_FT)
    }
    with open(out_dir / "summary_training.yaml", 'w') as f:
        yaml.dump(summary_data, f)
    
    try:
        plot_training_curves(history, out_dir, epochs_tl, class_names)
    except Exception as e:
        print(f"⚠️ Aviso: No se pudieron generar las gráficas ({e})")
        
    print(f"🎉 Entregable final generado en {out_dir}")
    
    # ==========================================
    # CLEANUP
    # =========================================
    del model, optimizer, scheduler, warmup_scheduler, train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("🧹 Memoria limpiada, pipeline finalizado.")