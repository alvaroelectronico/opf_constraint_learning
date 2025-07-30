"""
Trainer específico para la red neuronal del módulo complejo.
"""

import torch
import numpy as np
from typing import Tuple
from common.base_trainer import BaseTrainer
from .model import ComplexModulusModel
from .config import ComplexModulusConfig


class ComplexModulusTrainer(BaseTrainer):
    """
    Trainer para la red neuronal del módulo complejo.
    """
    
    def __init__(self, model: ComplexModulusModel, config: ComplexModulusConfig):
        super().__init__(model, config)
    
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


def evaluate_model(model: ComplexModulusModel, x_re: float, x_im: float) -> Tuple[float, float]:
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