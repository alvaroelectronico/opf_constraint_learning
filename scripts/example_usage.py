"""
Ejemplo de uso de ambas redes neuronales del proyecto OPF.
"""

import torch
import numpy as np

from networks.complex_modulus import ComplexModulusModel, ComplexModulusConfig, evaluate_model as eval_complex
from networks.power_flow import PowerFlowModel, PowerFlowConfig, evaluate_model as eval_power_flow


def example_complex_modulus():
    """
    Ejemplo de uso de la red neuronal del módulo complejo.
    """
    print("=== Ejemplo: Red Neuronal de Módulo Complejo ===")
    
    # Configuración
    config = ComplexModulusConfig()
    
    # Crear modelo
    model = ComplexModulusModel(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Ejemplos de predicción (sin entrenar, serán aleatorias)
    test_cases = [
        (1.0, 0.0),    # |1| = 1
        (0.0, 1.0),    # |i| = 1
        (3.0, 4.0),    # |3+4i| = 5
        (-5.0, 12.0),  # |-5+12i| = 13
    ]
    
    print("\nPredicciones (modelo sin entrenar):")
    for x_re, x_im in test_cases:
        prediction, exact = eval_complex(model, x_re, x_im)
        print(f"|{x_re}+{x_im}i|: Predicción={prediction:.6f}, Exacto={exact:.6f}")
    
    print("\n" + "="*60)


def example_power_flow():
    """
    Ejemplo de uso de la red neuronal de flujo de potencia.
    """
    print("=== Ejemplo: Red Neuronal de Flujo de Potencia ===")
    
    # Configuración
    config = PowerFlowConfig()
    
    # Crear modelo
    model = PowerFlowModel(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Ejemplos de predicción (sin entrenar, serán aleatorias)
    test_cases = [
        (1.0, 0.0, 0.95, 0.0, 1.0, -2.0),    # Caso típico
        (1.0, 0.1, 0.9, 0.05, 0.5, -1.5),    # Con componentes imaginarias
        (0.95, 0.0, 1.0, 0.0, 2.0, -3.0),    # Alta conductancia
    ]
    
    print("\nPredicciones (modelo sin entrenar):")
    for u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij in test_cases:
        (pred_p, pred_q), (exact_p, exact_q) = eval_power_flow(model, u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij)
        print(f"Entrada: u_Ri={u_Ri}, u_Ii={u_Ii}, u_Rj={u_Rj}, u_Ij={u_Ij}, G_ij={G_ij}, B_ij={B_ij}")
        print(f"  P: Predicción={pred_p:.6f}, Exacto={exact_p:.6f}")
        print(f"  Q: Predicción={pred_q:.6f}, Exacto={exact_q:.6f}")
        print()
    
    print("="*60)


def example_batch_prediction():
    """
    Ejemplo de predicción en lote para ambas redes.
    """
    print("=== Ejemplo: Predicción en Lote ===")
    
    # Red de módulo complejo
    config_complex = ComplexModulusConfig()
    model_complex = ComplexModulusModel(
        input_size=config_complex.INPUT_SIZE,
        hidden_sizes=config_complex.HIDDEN_SIZES,
        output_size=config_complex.OUTPUT_SIZE
    )
    
    # Datos de entrada en lote
    batch_inputs = torch.FloatTensor([
        [1.0, 0.0],   # |1| = 1
        [0.0, 1.0],   # |i| = 1
        [3.0, 4.0],   # |3+4i| = 5
        [-5.0, 12.0], # |-5+12i| = 13
    ])
    
    # Predicción en lote
    model_complex.eval()
    with torch.no_grad():
        batch_predictions = model_complex(batch_inputs)
    
    print("Predicciones en lote - Módulo Complejo:")
    for i, (inputs, prediction) in enumerate(zip(batch_inputs, batch_predictions)):
        x_re, x_im = inputs.numpy()
        pred = prediction.item()
        exact = np.sqrt(x_re**2 + x_im**2)
        print(f"  {i+1}. |{x_re}+{x_im}i|: Predicción={pred:.6f}, Exacto={exact:.6f}")
    
    print("\n" + "="*60)


def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("Ejemplos de uso de las redes neuronales del proyecto OPF")
    print("="*60)
    
    # Ejemplo de módulo complejo
    example_complex_modulus()
    
    # Ejemplo de flujo de potencia
    example_power_flow()
    
    # Ejemplo de predicción en lote
    example_batch_prediction()
    
    print("\nNota: Los modelos mostrados no están entrenados.")
    print("Para obtener predicciones precisas, ejecuta los scripts de entrenamiento:")
    print("  - python networks/complex_modulus/train.py")
    print("  - python networks/power_flow/train.py")


if __name__ == "__main__":
    main() 