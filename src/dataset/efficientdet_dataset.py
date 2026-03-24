import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoDataset(Dataset):
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (str): Ruta al archivo .json en formato COCO.
            images_dir (str): Directorio donde están las imágenes.
            transform (albumentations.Compose): Lista de transformaciones.
        """
        super().__init__()
        self.coco = COCO(json_path)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # 1. Obtener ID e información de la imagen
        image_id = self.ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        img_w = image_info['width']
        img_h = image_info['height']
        image_path = image_info['file_name']
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # Normalización básica

        # 3. Cargar anotaciones
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        coco_anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in coco_anns:
            x, y, w, h = ann['bbox']
            
            # --- SOLUCIÓN: CLAMPING / CLIPPING ---
            # Aseguramos que x_min y y_min no sean negativos
            x1 = max(0, x)
            y1 = max(0, y)
            
            # Aseguramos que x_max y y_max no superen el tamaño de la imagen
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            # Recalculamos ancho y alto finales
            final_w = x2 - x1
            final_h = y2 - y1

            # Solo añadimos la caja si sigue siendo válida después del recorte
            if final_w > 0 and final_h > 0:
                boxes.append([x1, y1, final_w, final_h])
                labels.append(ann['category_id'])

        # 4. Aplicar transformaciones (Albumentations)
        if self.transform:
            sample = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)

        # 5. Convertir a NumPy para manipular coordenadas y evitar TypeErrors
        boxes = np.array(boxes)
        num_objs = len(boxes)

        if num_objs > 0:
            # Transformar de [x, y, w, h] -> [x1, y1, x2, y2]
            # Usamos una copia para evitar errores de referencia
            res_boxes = boxes.copy()
            res_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # x2 = x + w
            res_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # y2 = y + h
            
            # Transformar a formato EfficientDet: [y1, x1, y2, x2]
            boxes = res_boxes[:, [1, 0, 3, 2]]
        else:
            # Imagen sin objetos
            boxes = np.zeros((0, 4))

        # 6. Preparar el diccionario target
        target = {
            'bbox': torch.as_tensor(boxes, dtype=torch.float32),
            'cls': torch.as_tensor(labels, dtype=torch.int64),
            'img_id': torch.tensor([image_id]),
            'img_size': torch.tensor([(image_info['height'], image_info['width'])])
        }

        return image, target

def collate_fn(batch):
    """
    Función para empaquetar lotes con distinto número de objetos por imagen.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Definir transformaciones mínimas
    transforms = A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    json_path = '/home/carlo/MURIA_wsl/Reconocimient_Objetos/ObjectRecognition/dataset/processed/experiments/set1_balanced_subsampled/fold_1_train_coco.json'

    # Crear dataset
    dataset = CocoDataset(
        json_path=json_path, 
        images_dir='', # Vacío si las rutas son absolutas
        transform=transforms
    )

    # Crear loader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Probar un batch
    img_batch, target_batch = next(iter(loader))
    print(f"Batch de imágenes: {img_batch.shape}")
    print(f"Ejemplo de cajas en la primera imagen: {target_batch[0]['bbox']}")