"""
Red neuronal para aproximar el módulo de números complejos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import os


class ComplexModulusNN(nn.Module):
    """
    Red neuronal que aproxima el módulo de un número complejo.
    
    Entrada: [x_re, x_im] (partes real e imaginaria)
    Salida: |x| = sqrt(x_re^2 + x_im^2)
    """
    
    def __init__(self, input_size: int = 2, hidden_sizes: list = [64, 32, 16], output_size: int = 1):
        super(ComplexModulusNN, self).__init__()
        
        # Construir capas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red neuronal.
        
        Args:
            x: Tensor de forma (batch_size, 2) con [x_re, x_im]
            
        Returns:
            Tensor de forma (batch_size, 1) con el módulo aproximado
        """
        return self.network(x)


class ComplexModulusTrainer:
    """
    Clase para entrenar la red neuronal del módulo complejo.
    """
    
    def __init__(self, model: ComplexModulusNN, config):
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
        Genera datos de entrenamiento para el módulo complejo.
        
        Returns:
            Tuple con (inputs, targets) donde:
            - inputs: (num_samples, 2) con [x_re, x_im]
            - targets: (num_samples, 1) con |x|
        """
        # Generar números complejos aleatorios
        x_re = np.random.uniform(self.config.MIN_VALUE, self.config.MAX_VALUE, self.config.NUM_SAMPLES)
        x_im = np.random.uniform(self.config.MIN_VALUE, self.config.MAX_VALUE, self.config.NUM_SAMPLES)
        
        # Calcular el módulo exacto
        modulus = np.sqrt(x_re**2 + x_im**2)
        
        # Convertir a tensores
        inputs = torch.FloatTensor(np.column_stack([x_re, x_im]))
        targets = torch.FloatTensor(modulus).unsqueeze(1)
        
        return inputs, targets
    
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
    
    def train(self, train_loader, val_loader) -> dict:
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
    
    def save_model(self, path: str):
        """
        Guarda el modelo entrenado con información completa de configuración.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Información detallada de la arquitectura
        model_info = {
            'input_size': self.config.INPUT_SIZE,
            'hidden_sizes': self.config.HIDDEN_SIZES,
            'output_size': self.config.OUTPUT_SIZE,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
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
        
        # Información de datos
        data_info = {
            'num_samples': self.config.NUM_SAMPLES,
            'min_value': self.config.MIN_VALUE,
            'max_value': self.config.MAX_VALUE,
            'data_range': f"[{self.config.MIN_VALUE}, {self.config.MAX_VALUE}]"
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
            'data_info': data_info,
            'device_info': device_info,
            'save_timestamp': str(torch.datetime.now()) if hasattr(torch, 'datetime') else None
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


def exact_modulus(x_re: float, x_im: float) -> float:
    """
    Calcula el módulo exacto de un número complejo.
    
    Args:
        x_re: Parte real
        x_im: Parte imaginaria
        
    Returns:
        Módulo del número complejo
    """
    return np.sqrt(x_re**2 + x_im**2)


    def evaluate_model(self, model: ComplexModulusNN, x_re: float, x_im: float) -> Tuple[float, float]:
        """
        Evalúa el modelo en un punto específico.
        
        Args:
            model: Modelo entrenado
            x_re: Parte real
            x_im: Parte imaginaria
            
        Returns:
            Tuple con (predicción, valor_exacto)
        """
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor([[x_re, x_im]])
            prediction = model(input_tensor).item()
        
        exact_value = exact_modulus(x_re, x_im)
        return prediction, exact_value


def evaluate_model(model: ComplexModulusNN, x_re: float, x_im: float) -> Tuple[float, float]:
    """
    Evalúa el modelo en un punto específico.
    
    Args:
        model: Modelo entrenado
        x_re: Parte real
        x_im: Parte imaginaria
        
    Returns:
        Tuple con (predicción, valor_exacto)
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor([[x_re, x_im]])
        prediction = model(input_tensor).item()
    
    exact_value = exact_modulus(x_re, x_im)
    return prediction, exact_value 