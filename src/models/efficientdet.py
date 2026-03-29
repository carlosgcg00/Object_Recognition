import torch
from effdet import create_model, DetBenchTrain, DetBenchPredict

class EfficientDetModel:
    def __init__(self, model_name='efficientdet_d0', num_classes=80, image_size=(512, 512), pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.pretrained = pretrained
        
        # 1. Crear el modelo base (Backbone + BiFPN + Head)
        self.model = create_model(
            self.model_name,
            bench_task='',
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            image_size=self.image_size
        )

    def get_train_model(self, device=None):
        """Devuelve el modelo preparado para entrenamiento (calcula la pérdida)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_bench = DetBenchTrain(self.model).to(device)
        return train_bench

    def get_predict_model(self, checkpoint_path=None, device=None):
        """Devuelve el modelo preparado para inferencia (hace NMS y decodificación)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path:
            # Cargar pesos si tenemos un archivo .pth
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
        predict_bench = DetBenchPredict(self.model).to(device)
        predict_bench.eval() # Modo evaluación por defecto
        return predict_bench
        
def get_train_bench(num_classes=3, image_size=(512, 512), device='cpu'):
    detector = EfficientDetModel(num_classes=num_classes, image_size=image_size)
    return detector.get_train_model(device)

def get_predict_model(checkpoint_path, num_classes=3, image_size=(512, 512), device='cpu'):
    detector = EfficientDetModel(num_classes=num_classes, image_size=image_size, pretrained=False)
    return detector.get_predict_model(checkpoint_path, device)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Supongamos que tienes 2 clases: 'perro' y 'gato'
    my_detector = EfficientDetModel(model_name='efficientdet_d1', num_classes=2)

    # Para ENTRENAR
    model_train = my_detector.get_train_model()
    print("Modelo listo para el loop de entrenamiento.")

    # Para INFERENCIA
    model_eval = my_detector.get_predict_model()
    print("Modelo listo para detectar objetos.")