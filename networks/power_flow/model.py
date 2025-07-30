"""
Modelo específico para la red neuronal de flujo de potencia.
"""

from common.base_model import BaseModel


class PowerFlowModel(BaseModel):
    """
    Red neuronal que estima las potencias activa y reactiva entre dos nodos.
    
    Entrada: [u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij]
    - u_Ri, u_Ii: Componentes real e imaginaria de la tensión en el nodo i
    - u_Rj, u_Ij: Componentes real e imaginaria de la tensión en el nodo j
    - G_ij: Conductancia de la línea i-j
    - B_ij: Susceptancia de la línea i-j
    
    Salida: [p_ij, q_ij]
    - p_ij: Potencia activa de i a j
    - q_ij: Potencia reactiva de i a j
    """
    
    def __init__(self, input_size: int = 6, hidden_sizes: list = [128, 64, 32], output_size: int = 2):
        super().__init__(input_size, hidden_sizes, output_size)
    
    def forward(self, x):
        """
        Forward pass de la red neuronal.
        
        Args:
            x: Tensor de forma (batch_size, 6) con [u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij]
            
        Returns:
            Tensor de forma (batch_size, 2) con [p_ij, q_ij]
        """
        return super().forward(x) 