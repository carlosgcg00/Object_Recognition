# src/models/efficientdet.py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
from effdet import create_model, create_model_from_config, get_efficientdet_config, DetBenchTrain, DetBenchPredict
import math

class EfficientDet(nn.Module):
    """
    Wrapper for EfficientDet that automatically manages the 'Benches' 
    for training and prediction.
    """
    def __init__(
        self, 
        num_classes: int, 
        img_size: int = 640, 
        architecture: str = 'tf_efficientdet_d1',
        is_training: bool = True,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.architecture = architecture
        self.is_training = is_training

        # We create the base model and wrap it in the corresponding Bench
        # 'train' -> DetBenchTrain (returns losses)
        # 'predict' -> DetBenchPredict (returns NMS detections)
        bench_task = 'train' if self.is_training else 'predict'
        
        # Load config of the model
        config = get_efficientdet_config(self.architecture)
        
        config.fpn_drop_path_rate = dropout_rate
        
        # self.model = create_model(
        #     model_name=self.architecture,
        #     bench_task=bench_task,
        #     num_classes=self.num_classes,
        #     pretrained=pretrained, 
        #     pretrained_backbone=True,
        #     image_size=(self.img_size, self.img_size),
        #     bench_labeler=True,
        #     )
        
        self.model = create_model_from_config(
            config,
            bench_task=bench_task,
            num_classes=self.num_classes,
            pretrained=pretrained, 
            pretrained_backbone=True,
            image_size=(self.img_size, self.img_size),
            bench_labeler=True
        )
        
    def forward(self, images: torch.Tensor, targets: Optional[dict] = None):
        """
        If is_training=True, expects targets and returns a dictionary of losses.
        If is_training=False, returns detection tensors [x1, y1, x2, y2, score, class].
        
        Args:
            images (torch.Tensor): Batch of images.
            targets (Optional[dict]): Batch of targets (only in training mode).
        
        Returns:
            dict or torch.Tensor: Dictionary of losses (training) or detection tensors (inference).
        """
        if self.is_training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def switch_to_predict(self):
        """ Changes the model to Inference mode (NMS) maintaining the current weights.
        
        Args:
            None
        
        Returns:
            None
        """
        if self.is_training:
            device = next(self.model.parameters()).device
            # We extract the base model from the training bench
            base_model = self.model.model 
            self.model = DetBenchPredict(base_model).to(device)
            self.is_training = False
            self.eval()
    
    def switch_to_eval(self):
        """ Changes the model to Evaluation mode (NMS) maintaining the current weights.
        
        Args:
            None
        
        Returns:
            None
        """
        if self.is_training:
            device = next(self.model.parameters()).device
            # We extract the base model from the training bench
            base_model = self.model.model 
            self.model = DetBenchPredict(base_model).to(device)
            self.is_training = False
            self.eval()

    def switch_to_train(self):
        """ Changes the model to Training mode (Loss) maintaining the current weights.
        
        Args:
            None
        
        Returns:
            None
        """
        if not self.is_training:
            device = next(self.model.parameters()).device
            # We extract the base model from the prediction bench
            base_model = self.model.model
            self.model = DetBenchTrain(base_model).to(device)
            self.is_training = True
            self.train()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('model.', '').replace('base_model.', '').replace('module.', '')
            new_state_dict[name] = v
            
        # --- NUEVO: Borrar las anclas cacheadas que el modelo base no necesita ---
        if "anchors.boxes" in new_state_dict:
            del new_state_dict["anchors.boxes"]
            
        # Inyectamos en la red base
        self.model.model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Weights loaded successfully from {checkpoint_path.name}") 

    def save_checkpoint(self, checkpoint_path: Union[str, Path]):
        """ Saves the model weights to a checkpoint.
        
        Args:
            checkpoint_path (Union[str, Path]): The path to save the checkpoint to.
        
        Returns:
            None
        """
        checkpoint_path = Path(checkpoint_path)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✅ Weights saved successfully to {checkpoint_path.name}")

    def freeze_backbone(self):
        """ Freezes the backbone of the model except the head. """
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the head
        for param in self.model.model.class_net.parameters():
            param.requires_grad = True
        for param in self.model.model.box_net.parameters():
            param.requires_grad = True
        print("✅ Backbone frozen, head unfrozen")

    def unfreeze_all(self):
        """ Unfreezes all the parameters of the model. """
        for param in self.model.parameters():
            param.requires_grad = True
        print("✅ All parameters unfrozen")

    def get_trainable_params(self):
        """ Returns the number of trainable parameters. """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_size_mb(self):
        """Calcula el tamaño del modelo en memoria RAM (MB)."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def get_total_params(self):
        """Calcula el número total de parámetros del modelo."""
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params_percentage(self):
        """Calcula el porcentaje de parámetros entrenables."""
        total_params = self.get_total_params()
        trainable_params = self.get_trainable_params()
        return (trainable_params / total_params) * 100

    def get_model_info(self):
        """Devuelve un diccionario con información detallada del modelo."""
        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "is_training": self.is_training,
            "total_params": self.get_total_params(),
            "trainable_params": self.get_trainable_params(),
            "trainable_params_percentage": round(self.get_trainable_params_percentage(), 2),
            "model_size_GB": round(self.get_model_size_mb() / 1024, 2)
        }
        