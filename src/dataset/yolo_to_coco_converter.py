# src/dataset/yolo_to_coco_converter.py
import json
import cv2
from pathlib import Path
from typing import List

def convert_yolo_split_to_coco(
    yolo_split_txt: Path, 
    output_json: Path, 
    class_names: List[str]
) -> None:
    """
    Convierte un archivo .txt de partición YOLO en un archivo .json formato COCO.
    
    Args:
        yolo_split_txt (Path): Ruta al archivo .txt (ej. fold_1_train.txt).
        output_json (Path): Ruta donde se guardará el .json resultante.
        class_names (List[str]): Lista de nombres de las clases en orden (ID 0, 1, 2).
    """
    # 1. Estructura base de COCO
    coco_data = {
        "info": {"description": "Dataset exportado de YOLO a COCO para el Master"},
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
        "images": [],
        "annotations": []
    }
    
    # 2. Leer las rutas de las imágenes desde el .txt de YOLO
    with open(yolo_split_txt, 'r', encoding='utf-8') as f:
        image_paths = [Path(line.strip()) for line in f.readlines() if line.strip()]
        
    annotation_id = 1 # COCO requiere un ID único para cada caja delimitadora
    
    for image_id, img_path in enumerate(image_paths, start=1):
        # Leemos la imagen solo para obtener su ancho y alto real
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: No se pudo leer la imagen {img_path}")
            continue
            
        height, width, _ = img.shape
        
        # Añadimos la información de la imagen al JSON
        coco_data["images"].append({
            "id": image_id,
            "file_name": str(img_path.resolve()), # Ruta absoluta
            "width": width,
            "height": height
        })
        
        # 3. Aplicamos el truco de Ultralytics para encontrar la etiqueta
        lbl_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt'))
        
        if lbl_path.exists():
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    class_id = int(parts[0])
                    x_c, y_c, w_n, h_n = map(float, parts[1:5])
                    
                    # 4. Desnormalizamos las coordenadas (YOLO a COCO)
                    # COCO usa [x_min, y_min, ancho_absoluto, alto_absoluto]
                    w_px = w_n * width
                    h_px = h_n * height
                    x_min = (x_c * width) - (w_px / 2.0)
                    y_min = (y_c * height) - (h_px / 2.0)
                    
                    # Añadimos la anotación
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)],
                        "area": round(w_px * h_px, 2),
                        "iscrowd": 0 # 0 indica que es un objeto individual, no una multitud
                    })
                    annotation_id += 1
                    
    # 5. Guardar el JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"✅ Formato COCO generado con éxito: {output_json.name}")