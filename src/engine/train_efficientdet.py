# src/engine/train_efficientdet.py

import torch
from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from models.efficientdet import get_predict_model

def train_one_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train.
        loader: The data loader.
        optimizer: The optimizer.
        device: The device to train on.
    
    Returns:
        The average loss for the epoch.
    """
    model.train()
    runing_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, targets in pbar:
        # Torch.stack() is used to stack the images along a new dimension.
        # .to(device) is used to move the images to the device.
        # .float() is used to convert the images to float.
        if isinstance(images, list):
            images = torch.stack(images).to(device).float()
        else:
            images = images.to(device).float()
        
        # Targets are a list of dictionaries, each containing 'bbox' and 'cls'.
        # We need to move them to the device.
        target_res = {
            'bbox': [t['bbox'].to(device) for t in targets],
            'cls': [t['cls'].to(device) for t in targets]
        }
        
        optimizer.zero_grad() # Reset the gradients.
        
        # Forward pass
        outputs = model(images, target_res)
        
        # Loss
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        runing_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return runing_loss / len(loader)
        
def validate_one_epoch(model, loader, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: The model to validate.
        loader: The data loader.
        device: The device to validate on.
    
    Returns:
        The average loss for the epoch.
    """
    model.eval()
    runing_loss = 0.0
    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, targets in pbar:
            # Torch.stack() is used to stack the images along a new dimension.
            # .to(device) is used to move the images to the device.
            # .float() is used to convert the images to float.
            if isinstance(images, list):
                images = torch.stack(images).to(device).float()
            else:
                images = images.to(device).float()
            
            batch_size = images.shape[0]
            h, w = images.shape[2], images.shape[3] # [B,3,H,W]

            # Prepare Boxes
            max_boxes = max([t['bbox'].shape[0] for t in targets])
            filled_boxes = torch.zeros((batch_size, max_boxes, 4), device=device)
            filled_labels = torch.zeros((batch_size, max_boxes), device=device) - 1

            for i, t in enumerate(targets):
                num_boxes = t['bbox'].shape[0]
                if num_boxes > 0:
                    filled_boxes[i, :num_boxes] = t['bbox'].to(device)
                    filled_labels[i, :num_boxes] = t['cls'].to(device)
            
            # Targets are a list of dictionaries, each containing 'bbox' and 'cls'.
            # We need to move them to the device.
            target_res = {
                'bbox': filled_boxes,
                'cls': filled_labels,
                'img_size': torch.tensor([(h, w)] * batch_size, device=device),
                'img_scale': torch.ones(batch_size, device=device)
            }
            
            # Forward pass
            outputs = model(images, target_res)
            
            # Loss
            loss = outputs['loss']
            
            runing_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    return runing_loss / len(loader)


def evaluate_test_set(model_path, test_loader, device, num_classes=3):
    """
    Evaluate a pt model.
    
    Args:
        model_path: Path to the trained model.
        test_loader: The test data loader.
        device: The device to evaluate on.
        num_classes: The number of classes.
    
    Returns:
        A dictionary containing the evaluation results.
    """    
    model = get_predict_model(model_path, num_classes, device=device)
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            # --- EL FIX AQUÍ ---
            if isinstance(images, (list, tuple)):
                images = torch.stack(images).to(device).float()
            else:
                images = images.to(device).float()
            
            outputs = model(images)
            preds = []
            for res in outputs:
                mask = res[:, 4] > 0.05
                preds.append({
                    'boxes': res[mask, :4].detach().cpu(),
                    'scores': res[mask, 4].detach().cpu(),
                    'labels': res[mask, 5].detach().cpu().to(torch.int64)
                })

            gts = []
            for t in targets:
                boxes_xyxy = t['bbox'][:, [1, 0, 3, 2]]
                gts.append({
                    'boxes': boxes_xyxy.detach().cpu(),
                    'labels': t['cls'].detach().cpu().to(torch.int64)
                })
            
            metric.update(preds, gts)
    
    return metric.compute()

def train_efficientdet(model, optimizer, train_loader, val_loader, test_loader, num_epochs, scheduler = None, num_classes = 3, folder_path = None, device = None):
    """
    Train the model for a specified number of epochs.
    
    Args:
        model: The model to train.
        optimizer: The optimizer.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        test_loader: The test data loader.
        num_epochs: The number of epochs to train for.
        scheduler: The learning rate scheduler.
        num_classes: The number of classes.
        folder_path: The path to save the model checkpoints.
        device: The device to train on.
    
    Returns:
        summary_dict: A dictionary containing the training history and last and best model checkpoint paths and test metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary_dict = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'best_model_path': None,
        'best_test_metrics': None,
        'last_model_path': None,
        'last_test_metrics': None        
    }

    best_val_loss = float('inf')
    progress_bar = tqdm(range(num_epochs), desc="Training")
    for epoch in progress_bar:
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        summary_dict['train_loss'].append(train_loss)
        summary_dict['val_loss'].append(val_loss)
        summary_dict['learning_rate'].append(optimizer.param_groups[0]['lr'])
        if scheduler is not None:
            scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            summary_dict['best_model_path'] = f'{folder_path}/best_model_epoch_{epoch}.pth'
            torch.save(model.model.state_dict(), summary_dict['best_model_path'])

    summary_dict['last_model_path'] = f'{folder_path}/last_model_epoch_{epoch}.pth'
    torch.save(model.model.state_dict(), summary_dict['last_model_path'])
        
    if summary_dict['best_model_path'] is not None:
        summary_dict['best_test_metrics'] = evaluate_test_set(summary_dict['best_model_path'], test_loader, device, num_classes)
    if summary_dict['last_model_path'] is not None:
        summary_dict['last_test_metrics'] = evaluate_test_set(summary_dict['last_model_path'], test_loader, device, num_classes)
    print(f"Training completed. Best model saved to {summary_dict['best_model_path']}")
    print(f"Last model saved to {summary_dict['last_model_path']}")
    return summary_dict
    