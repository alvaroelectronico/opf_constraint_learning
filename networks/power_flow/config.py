"""
Configuración para la red neuronal de flujo de potencia.
"""

from common.config import BaseConfig


class PowerFlowConfig(BaseConfig):
    """
    Configuración para la red neuronal que estima potencias activa y reactiva.
    
    Basado en las ecuaciones:
    p_ij = G_ij * (u_Ri^2 + u_Ii^2 - u_Ri*u_Rj - u_Ii*u_Ij) + B_ij * (u_Ri*u_Ij - u_Ii*u_Rj)
    q_ij = -B_ij * (u_Ri^2 + u_Ii^2 - u_Ri*u_Rj - u_Ii*u_Ij) + G_ij * (u_Ri*u_Ij - u_Ii*u_Rj)
    """
    
    def __init__(self):
        super().__init__("power_flow")
        
        # Parámetros específicos de la red neuronal
        self.INPUT_SIZE = 6  # u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij
        self.HIDDEN_SIZES = [128, 64, 32]  # Capas ocultas más grandes para mayor complejidad
        self.OUTPUT_SIZE = 2  # p_ij, q_ij
        
        # Parámetros específicos de generación de datos
        self.VOLTAGE_MIN = 0.8  # Mínimo valor de tensión (pu)
        self.VOLTAGE_MAX = 1.2  # Máximo valor de tensión (pu)
        self.CONDUCTANCE_MIN = -2.0  # Mínimo valor de conductancia
        self.CONDUCTANCE_MAX = 2.0   # Máximo valor de conductancia
        self.SUSCEPTANCE_MIN = -5.0   # Mínimo valor de susceptancia
        self.SUSCEPTANCE_MAX = 5.0    # Máximo valor de susceptancia 