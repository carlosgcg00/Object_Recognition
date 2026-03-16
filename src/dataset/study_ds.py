# src/dataset/study_ds.py
# Import utils
from utils.utils import read_yaml,load_video, load_gt, format_gt

# Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent
PATHS_FILE = PROJECT_ROOT / "paths.yaml"

PATHS = read_yaml(PATHS_FILE)

def get_dataset_paths():
    dataset_root = PROJECT_ROOT / PATHS["dataset"]["path"]
    classes = PATHS["dataset"]["classes"]
    
    # Preparamos la estructura base que quieres
    paths = {
        "dataset_path": dataset_root,
        "labels": classes,
        "data": {}  # Usaremos una llave 'data' para agrupar las clases
    }

    for class_name in classes:
        # IMPORTANTE: Definir rutas relativas a la clase en cada iteración
        # Si usas videos_path = videos_path / ... dentro del bucle, la ruta crece infinitamente
        v_subpath = PATHS["dataset"]["videos"]["path"]
        g_subpath = PATHS["dataset"]["gt"]["path"]
        
        current_videos_path = dataset_root / class_name / v_subpath
        current_gt_path = dataset_root / class_name / g_subpath

        v_ext = PATHS["dataset"]["videos"]["extension"]
        g_ext = PATHS["dataset"]["gt"]["extension"]

        video_files = sorted(current_videos_path.glob(f"*{v_ext}"))
        
        pairs = []
        for v in video_files:
            # Construimos el nombre del label basado en el video
            label_path = current_gt_path / f"{v.stem}_gt{g_ext}"
            
            pairs.append({
                "video": str(v), # Convertimos a string para que sea más legible
                "label": str(label_path)
            })

        paths["data"][class_name] = pairs

    return paths

def extract_video_info(video_path):
    """Extract width, height, fps and frame_count of a video
    Args:
        video_path (str): path of the video
    Return:
        width: width of the video
        height: height of the video
        fps: frames per second
        frame_count: total number of frames
    """
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info

def summary_df(ds_info):
    """
    Genera un DataFrame resumen con información de videos y conteo de etiquetas por frame.
    """
    all_data_list = []

    for animal in ds_info['labels']:
        # Accedemos a la lista de pares (video, label) para cada animal
        for pairs in ds_info['data'][animal]:
            video_path = pairs['video']
            gt_path = pairs['label']
            
            # Obtener nombre del video y metadatos
            name_video = os.path.splitext(os.path.basename(video_path))[0]
            v_info = extract_video_info(video_path)
            
            # Cargar anotaciones usando tus funciones de utils.py
            raw_annotations = load_gt(gt_path)
            formatted_annotations = format_gt(raw_annotations)
            
            # Crear DataFrame temporal para contar etiquetas por frame_id
            df_temp = pd.DataFrame(formatted_annotations)
            
            if not df_temp.empty:
                # Agrupamos por frame_id para obtener el 'count'
                df_counts = df_temp.groupby('frame_id').size().reset_index(name='count')
                
                # Inyectamos los metadatos del video (se repiten por cada frame del mismo video)
                df_counts['video_name'] = name_video
                df_counts['class'] = animal
                df_counts['width'] = v_info['width']
                df_counts['height'] = v_info['height']
                df_counts['total_frames'] = v_info['frames']
                df_counts['fps'] = v_info['fps']
                
                all_data_list.append(df_counts)

    # Concatenamos todos los resultados en un solo DataFrame
    if all_data_list:
        df_final = pd.concat(all_data_list, ignore_index=True)
        # Reordenamos columnas para que sea "bonito" y cómodo de leer
        column_order = ['video_name', 'class', 'frame_id', 'count', 'width', 'height', 'total_frames', 'fps']
        return df_final[column_order]
    else:
        return pd.DataFrame() # Devuelve un DF vacío si no hay datos


if __name__ == "__main__":
    from pprint import pprint
    resultado = get_dataset_paths(PROJECT_ROOT, PATHS)
    pprint(resultado, sort_dicts=False, indent=2)