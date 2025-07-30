"""
Configuración específica para la red neuronal del módulo complejo.
"""

from common.config import BaseConfig


class ComplexModulusConfig(BaseConfig):
    """
    Configuración para la red neuronal que aproxima el módulo de números complejos.
    """
    
    def __init__(self):
        super().__init__("complex_modulus")
        
        # Parámetros específicos de la red neuronal
        self.INPUT_SIZE = 2  # x_re, x_im
        self.HIDDEN_SIZES = [64, 32, 16]  # Capas ocultas
        self.OUTPUT_SIZE = 1  # |x|
        
        # Parámetros específicos de generación de datos
        self.MIN_VALUE = -10.0
        self.MAX_VALUE = 10.0 