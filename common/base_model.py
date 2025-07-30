"""
Clase base para todos los modelos de redes neuronales del proyecto OPF.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class BaseModel(nn.Module):
    """
    Clase base para modelos de redes neuronales.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.1):
        """
        Inicializa el modelo base.
        
        Args:
            input_size: Tamaño de la entrada
            hidden_sizes: Lista con tamaños de las capas ocultas
            output_size: Tamaño de la salida
            dropout_rate: Tasa de dropout
        """
        super(BaseModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Construir capas
        self.layers = self._build_layers()
        
    def _build_layers(self) -> nn.Sequential:
        """
        Construye las capas de la red neuronal.
        
        Returns:
            Secuencia de capas
        """
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, self.output_size))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red neuronal.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        return self.layers(x)
    
    def get_parameter_count(self) -> int:
        """
        Obtiene el número total de parámetros.
        
        Returns:
            Número de parámetros
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """
        Obtiene el número de parámetros entrenables.
        
        Returns:
            Número de parámetros entrenables
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': self.get_parameter_count(),
            'trainable_parameters': self.get_trainable_parameter_count()
        } 