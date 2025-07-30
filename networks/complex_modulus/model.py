"""
Modelo específico para la red neuronal del módulo complejo.
"""

from common.base_model import BaseModel


class ComplexModulusModel(BaseModel):
    """
    Red neuronal que aproxima el módulo de un número complejo.
    
    Entrada: [x_re, x_im] (partes real e imaginaria)
    Salida: |x| = sqrt(x_re^2 + x_im^2)
    """
    
    def __init__(self, input_size: int = 2, hidden_sizes: list = [64, 32, 16], output_size: int = 1):
        super().__init__(input_size, hidden_sizes, output_size)
    
    def forward(self, x):
        """
        Forward pass de la red neuronal.
        
        Args:
            x: Tensor de forma (batch_size, 2) con [x_re, x_im]
            
        Returns:
            Tensor de forma (batch_size, 1) con el módulo aproximado
        """
        return super().forward(x) 