import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mode: str, target_size=512):
    """
    Get the transforms for the given mode.
    
    Args:
        mode (str): The mode to get the transforms for. Can be 'train', 'val', or 'test'.
        target_size (tuple): The target size of the images.
        
    Returns:
        A.Compose: The transforms for the given mode.
    """
    if mode == 'train':
        return A.Compose([
            # 1. Recorte aleatorio (50% de probabilidad)
            A.RandomResizedCrop(
                size=(target_size, target_size), 
                scale=(0.5, 1.0), 
                p=0.5
            ),
            
            A.HorizontalFlip(p=0.5),
            
            A.CoarseDropout(
                max_holes=3, 
                max_height=int(target_size * 0.05), 
                max_width=int(target_size * 0.05), 
                p=0.1
            ),
            
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
            
            # --- EL SEGURO DE VIDA ---
            # Si RandomResizedCrop NO se activó, esta línea asegura que 
            # la imagen sea 512x512 sí o sí antes de salir.
            A.Resize(height=target_size, width=target_size, p=1.0), 
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels'], min_area=1, min_visibility=0.1))
    
    elif mode in ['val', 'test']:
        return A.Compose([
            A.Resize(height=target_size, width=target_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
        
    else:
        raise ValueError(f"Fase no reconocida: {mode}")