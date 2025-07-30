"""
Ejemplo simple de uso de la red neuronal para módulo complejo.
"""

import torch
import numpy as np
from complex_modulus_nn import ComplexModulusNN, exact_modulus
from config import Config


def simple_example():
    """
    Ejemplo simple de uso del modelo.
    """
    print("=== Ejemplo Simple: Módulo de Números Complejos ===")
    
    # Crear modelo
    config = Config()
    model = ComplexModulusNN(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Ejemplos de números complejos
    test_cases = [
        (3.0, 4.0),    # |3+4i| = 5
        (1.0, 1.0),    # |1+i| = √2 ≈ 1.414
        (0.0, 5.0),    # |5i| = 5
        (-3.0, 0.0),   # |-3| = 3
    ]
    
    print("\nEjemplos de cálculo del módulo:")
    print(f"{'Número':<12} {'Módulo Exacto':<15} {'Módulo NN':<15} {'Error':<10}")
    print("-" * 55)
    
    for x_re, x_im in test_cases:
        # Cálculo exacto
        exact = exact_modulus(x_re, x_im)
        
        # Predicción de la red neuronal (sin entrenar, será aleatoria)
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor([[x_re, x_im]])
            prediction = model(input_tensor).item()
        
        error = abs(prediction - exact)
        
        # Formatear el número complejo
        if x_im >= 0:
            complex_str = f"{x_re}+{x_im}i"
        else:
            complex_str = f"{x_re}{x_im}i"
        
        print(f"{complex_str:<12} {exact:<15.6f} {prediction:<15.6f} {error:<10.6f}")
    
    print("\nNota: Las predicciones son aleatorias porque el modelo no está entrenado.")
    print("Ejecuta 'python train.py' para entrenar el modelo.")


def batch_example():
    """
    Ejemplo de procesamiento por lotes.
    """
    print("\n=== Ejemplo de Procesamiento por Lotes ===")
    
    # Crear modelo
    config = Config()
    model = ComplexModulusNN(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    # Crear un lote de números complejos
    batch_size = 5
    x_re_batch = torch.tensor([1.0, 0.0, 3.0, -2.0, 0.5])
    x_im_batch = torch.tensor([0.0, 1.0, 4.0, 3.0, 0.5])
    
    # Combinar en tensor de entrada
    inputs = torch.stack([x_re_batch, x_im_batch], dim=1)
    
    print(f"Lote de entrada ({batch_size} números complejos):")
    for i in range(batch_size):
        x_re, x_im = x_re_batch[i].item(), x_im_batch[i].item()
        if x_im >= 0:
            print(f"  {i+1}. {x_re}+{x_im}i")
        else:
            print(f"  {i+1}. {x_re}{x_im}i")
    
    # Procesar lote
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
    
    print(f"\nPredicciones del modelo:")
    for i in range(batch_size):
        x_re, x_im = x_re_batch[i].item(), x_im_batch[i].item()
        exact = exact_modulus(x_re, x_im)
        pred = predictions[i].item()
        error = abs(pred - exact)
        
        if x_im >= 0:
            complex_str = f"{x_re}+{x_im}i"
        else:
            complex_str = f"{x_re}{x_im}i"
        
        print(f"  |{complex_str}|: Predicción={pred:.6f}, Exacto={exact:.6f}, Error={error:.6f}")


def main():
    """
    Función principal del ejemplo.
    """
    simple_example()
    batch_example()
    
    print("\n=== Para entrenar el modelo ===")
    print("1. Instala las dependencias: pip install -r requirements.txt")
    print("2. Entrena el modelo: python train.py")
    print("3. Prueba el modelo: python test.py")


if __name__ == "__main__":
    main() 