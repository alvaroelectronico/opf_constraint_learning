"""
Trainer específico para la red neuronal de flujo de potencia.
"""

import torch
import numpy as np
from typing import Tuple
from common.base_trainer import BaseTrainer
from .model import PowerFlowModel
from .config import PowerFlowConfig


class PowerFlowTrainer(BaseTrainer):
    """
    Trainer para la red neuronal de flujo de potencia.
    """
    
    def __init__(self, model: PowerFlowModel, config: PowerFlowConfig):
        super().__init__(model, config)
    
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera datos de entrenamiento para el flujo de potencia.
        
        Returns:
            Tuple con (inputs, targets) donde:
            - inputs: (num_samples, 6) con [u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij]
            - targets: (num_samples, 2) con [p_ij, q_ij]
        """
        # Generar datos aleatorios
        u_Ri = np.random.uniform(self.config.VOLTAGE_MIN, self.config.VOLTAGE_MAX, self.config.NUM_SAMPLES)
        u_Ii = np.random.uniform(self.config.VOLTAGE_MIN, self.config.VOLTAGE_MAX, self.config.NUM_SAMPLES)
        u_Rj = np.random.uniform(self.config.VOLTAGE_MIN, self.config.VOLTAGE_MAX, self.config.NUM_SAMPLES)
        u_Ij = np.random.uniform(self.config.VOLTAGE_MIN, self.config.VOLTAGE_MAX, self.config.NUM_SAMPLES)
        G_ij = np.random.uniform(self.config.CONDUCTANCE_MIN, self.config.CONDUCTANCE_MAX, self.config.NUM_SAMPLES)
        B_ij = np.random.uniform(self.config.SUSCEPTANCE_MIN, self.config.SUSCEPTANCE_MAX, self.config.NUM_SAMPLES)
        
        # Calcular potencias exactas usando las ecuaciones
        p_ij, q_ij = self._calculate_exact_power_flow(u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij)
        
        # Convertir a tensores
        inputs = torch.FloatTensor(np.column_stack([u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij]))
        targets = torch.FloatTensor(np.column_stack([p_ij, q_ij]))
        
        return inputs, targets
    
    def _calculate_exact_power_flow(self, u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij):
        """
        Calcula las potencias exactas usando las ecuaciones de flujo de potencia.
        
        Args:
            u_Ri, u_Ii: Componentes de tensión en el nodo i
            u_Rj, u_Ij: Componentes de tensión en el nodo j
            G_ij: Conductancia
            B_ij: Susceptancia
            
        Returns:
            Tuple con (p_ij, q_ij)
        """
        # Términos comunes
        u_Ri_sq = u_Ri**2
        u_Ii_sq = u_Ii**2
        u_Ri_u_Rj = u_Ri * u_Rj
        u_Ii_u_Ij = u_Ii * u_Ij
        u_Ri_u_Ij = u_Ri * u_Ij
        u_Ii_u_Rj = u_Ii * u_Rj
        
        # Término común en ambas ecuaciones
        term1 = u_Ri_sq + u_Ii_sq - u_Ri_u_Rj - u_Ii_u_Ij
        term2 = u_Ri_u_Ij - u_Ii_u_Rj
        
        # Potencia activa: p_ij = G_ij * term1 + B_ij * term2
        p_ij = G_ij * term1 + B_ij * term2
        
        # Potencia reactiva: q_ij = -B_ij * term1 + G_ij * term2
        q_ij = -B_ij * term1 + G_ij * term2
        
        return p_ij, q_ij


def exact_power_flow(u_Ri: float, u_Ii: float, u_Rj: float, u_Ij: float, G_ij: float, B_ij: float) -> Tuple[float, float]:
    """
    Calcula las potencias exactas para un caso específico.
    
    Args:
        u_Ri, u_Ii: Componentes de tensión en el nodo i
        u_Rj, u_Ij: Componentes de tensión en el nodo j
        G_ij: Conductancia
        B_ij: Susceptancia
        
    Returns:
        Tuple con (p_ij, q_ij)
    """
    # Términos comunes
    term1 = u_Ri**2 + u_Ii**2 - u_Ri*u_Rj - u_Ii*u_Ij
    term2 = u_Ri*u_Ij - u_Ii*u_Rj
    
    # Potencia activa
    p_ij = G_ij * term1 + B_ij * term2
    
    # Potencia reactiva
    q_ij = -B_ij * term1 + G_ij * term2
    
    return p_ij, q_ij


def evaluate_model(model: PowerFlowModel, u_Ri: float, u_Ii: float, u_Rj: float, u_Ij: float, G_ij: float, B_ij: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Evalúa el modelo en un punto específico.
    
    Args:
        model: Modelo entrenado
        u_Ri, u_Ii: Componentes de tensión en el nodo i
        u_Rj, u_Ij: Componentes de tensión en el nodo j
        G_ij: Conductancia
        B_ij: Susceptancia
        
    Returns:
        Tuple con ((predicción_p, predicción_q), (exacto_p, exacto_q))
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor([[u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij]])
        prediction = model(input_tensor).squeeze().numpy()
        pred_p, pred_q = prediction[0], prediction[1]
    
    exact_p, exact_q = exact_power_flow(u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij)
    
    return (pred_p, pred_q), (exact_p, exact_q) 