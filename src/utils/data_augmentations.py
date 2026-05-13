import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(
        mode: str, 
        target_size=512, 
        random_crop_prob=0.5, 
        random_flip_prob=0.5, 
        random_coarse_dropout_prob=0.1, 
        random_color_jitter_prob=0.4
    ):
    """
    Get the transforms for the given mode.
    
    Args:
        mode (str): The mode to get the transforms for. Can be 'train', 'val', or 'test'.
        target_size (tuple): The target size of the images. 
        random_crop_prob (float): The probability of applying random crop.
        random_flip_prob (float): The probability of applying random flip.
        random_coarse_dropout_prob (float): The probability of applying random coarse dropout.
        random_color_jitter_prob (float): The probability of applying random color jitter.
        
    Returns:
        A.Compose: The transforms for the given mode.
    """
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(
                size=(target_size, target_size), 
                scale=(0.5, 1.0), 
                p=random_crop_prob
            ),
            
            A.HorizontalFlip(p=random_flip_prob),
            
            A.CoarseDropout(
                num_holes_range=(3, 5), 
                hole_height_range=(int(target_size * 0.05), int(target_size * 0.1)), 
                hole_width_range=(int(target_size * 0.05), int(target_size * 0.1)), 
                p=random_coarse_dropout_prob
            ),
            
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=random_color_jitter_prob),
            
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
        raise ValueError(f"Unknown phase: {mode}")