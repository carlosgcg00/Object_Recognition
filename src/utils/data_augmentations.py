import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(target_size=(512, 512)):
    return A.Compose([
        # 1. Aumentos de posición (cambian la caja)
        A.HorizontalFlip(p=0.5),
        A.RandomSizedCrop(
            min_max_height=(300, 512), 
            size=target_size,
            p=0.5
        ),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # 2. Aumentos de color/luz (no cambian la caja)
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        
        # 3. Paso final obligatorio
        A.Resize(target_size[0], target_size[1]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='coco',           # Le decimos que nuestras cajas entran como [x,y,w,h]
        label_fields=['labels']  # Le decimos dónde están las clases
    ))

def get_valid_transforms(target_size=(512, 512)):
    return A.Compose([
        A.Resize(target_size[0], target_size[1]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))