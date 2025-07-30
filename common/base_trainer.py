"""
Clase base para todos los trainers de redes neuronales del proyecto OPF.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
from .base_model import BaseModel
from .config import BaseConfig


class BaseTrainer:
    """
    Clase base para entrenar redes neuronales.
    """
    
    def __init__(self, model: BaseModel, config: BaseConfig):
        """
        Inicializa el trainer base.
        
        Args:
            model: Modelo a entrenar
            config: Configuración del entrenamiento
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizador y función de pérdida
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Historial de entrenamiento
        self.train_losses = []
        self.val_losses = []
        
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera datos de entrenamiento. Debe ser implementado por las subclases.
        
        Returns:
            Tuple con (inputs, targets)
        """
        raise NotImplementedError("Subclases deben implementar generate_data()")
    
    def train_epoch(self, train_loader) -> float:
        """
        Entrena la red por una época.
        
        Args:
            train_loader: DataLoader con datos de entrenamiento
            
        Returns:
            Pérdida promedio de la época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_inputs)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader) -> float:
        """
        Valida la red en el conjunto de validación.
        
        Args:
            val_loader: DataLoader con datos de validación
            
        Returns:
            Pérdida promedio de validación
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        Entrena la red neuronal completa.
        
        Args:
            train_loader: DataLoader con datos de entrenamiento
            val_loader: DataLoader con datos de validación
            
        Returns:
            Diccionario con el historial de entrenamiento
        """
        print(f"Entrenando en dispositivo: {self.device}")
        print(f"Tamaño del dataset: {len(train_loader.dataset)}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Entrenamiento
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validación
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Log cada 100 épocas
            if (epoch + 1) % 100 == 0:
                print(f"Época {epoch+1}/{self.config.NUM_EPOCHS}: "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, path: Optional[str] = None):
        """
        Guarda el modelo entrenado con información completa de configuración.
        
        Args:
            path: Ruta donde guardar el modelo (opcional)
        """
        if path is None:
            path = self.config.MODEL_PATH
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Información detallada de la arquitectura
        model_info = self.model.get_model_info()
        
        # Información de entrenamiento
        training_info = {
            'learning_rate': self.config.LEARNING_RATE,
            'batch_size': self.config.BATCH_SIZE,
            'num_epochs': self.config.NUM_EPOCHS,
            'train_split': self.config.TRAIN_SPLIT,
            'validation_split': self.config.VALIDATION_SPLIT,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'best_epoch': self.val_losses.index(min(self.val_losses)) if self.val_losses else None
        }
        
        # Información del dispositivo y optimizador
        device_info = {
            'device_used': str(self.device),
            'optimizer_type': type(self.optimizer).__name__,
            'criterion_type': type(self.criterion).__name__
        }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'model_info': model_info,
            'training_info': training_info,
            'device_info': device_info,
            'network_name': self.config.network_name
        }, path)
        
        print(f"Modelo guardado en: {path}")
        print(f"  - Parámetros totales: {model_info['total_parameters']:,}")
        print(f"  - Épocas entrenadas: {len(self.train_losses)}")
        print(f"  - Mejor pérdida de validación: {training_info['best_val_loss']:.6f}")
        print(f"  - Dispositivo usado: {device_info['device_used']}")
    
    def load_model(self, path: str):
        """
        Carga un modelo pre-entrenado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Modelo cargado desde: {path}")
    
    def evaluate_model(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evalúa el modelo con los inputs dados.
        
        Args:
            inputs: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
        return outputs 