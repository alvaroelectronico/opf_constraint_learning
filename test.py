"""
Script para probar y evaluar el modelo entrenado del módulo complejo.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from complex_modulus_nn import ComplexModulusNN, ComplexModulusTrainer, exact_modulus, evaluate_model
from config import Config


def load_trained_model(model_path: str):
    """
    Carga un modelo entrenado.
    
    Args:
        model_path: Ruta del modelo a cargar
        
    Returns:
        Tuple con (model, trainer, config)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', Config())
    
    # Crear modelo
    model = ComplexModulusNN(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Crear trainer
    trainer = ComplexModulusTrainer(model, config)
    trainer.load_model(model_path)
    
    return model, trainer, config


def test_specific_cases(model, trainer):
    """
    Prueba casos específicos conocidos.
    
    Args:
        model: Modelo entrenado
        trainer: Trainer del modelo
    """
    print("=== Pruebas de Casos Específicos ===")
    
    test_cases = [
        (0.0, 0.0, "Origen"),
        (1.0, 0.0, "Número real positivo"),
        (-1.0, 0.0, "Número real negativo"),
        (0.0, 1.0, "Unidad imaginaria"),
        (0.0, -1.0, "Unidad imaginaria negativa"),
        (3.0, 4.0, "Triángulo 3-4-5"),
        (-5.0, 12.0, "Triángulo 5-12-13"),
        (1.0, 1.0, "Diagonal del cuadrado"),
        (0.5, 0.5, "Diagonal del cuadrado pequeño"),
        (10.0, 0.0, "Número real grande"),
        (0.0, 10.0, "Número imaginario grande"),
    ]
    
    print(f"{'Caso':<25} {'x_re':<8} {'x_im':<8} {'Predicción':<12} {'Exacto':<12} {'Error':<10} {'Error %':<8}")
    print("-" * 85)
    
    for x_re, x_im, description in test_cases:
        prediction, exact = evaluate_model(model, x_re, x_im)
        error = abs(prediction - exact)
        error_percent = (error / (exact + 1e-8)) * 100
        
        print(f"{description:<25} {x_re:<8.2f} {x_im:<8.2f} {prediction:<12.6f} {exact:<12.6f} {error:<10.6f} {error_percent:<8.2f}%")


def test_random_samples(model, trainer, num_samples=1000, min_val=-10, max_val=10):
    """
    Prueba con muestras aleatorias.
    
    Args:
        model: Modelo entrenado
        trainer: Trainer del modelo
        num_samples: Número de muestras a probar
        min_val: Valor mínimo para las partes real e imaginaria
        max_val: Valor máximo para las partes real e imaginaria
    """
    print(f"\n=== Pruebas con {num_samples} Muestras Aleatorias ===")
    
    # Generar muestras aleatorias
    x_re_samples = np.random.uniform(min_val, max_val, num_samples)
    x_im_samples = np.random.uniform(min_val, max_val, num_samples)
    
    predictions = []
    exact_values = []
    errors = []
    relative_errors = []
    
    for x_re, x_im in zip(x_re_samples, x_im_samples):
        prediction, exact = evaluate_model(model, x_re, x_im)
        error = abs(prediction - exact)
        relative_error = (error / (exact + 1e-8)) * 100
        
        predictions.append(prediction)
        exact_values.append(exact)
        errors.append(error)
        relative_errors.append(relative_error)
    
    # Estadísticas
    predictions = np.array(predictions)
    exact_values = np.array(exact_values)
    errors = np.array(errors)
    relative_errors = np.array(relative_errors)
    
    print(f"Estadísticas de error:")
    print(f"  - Error absoluto promedio: {np.mean(errors):.6f}")
    print(f"  - Error absoluto máximo: {np.max(errors):.6f}")
    print(f"  - Error relativo promedio: {np.mean(relative_errors):.2f}%")
    print(f"  - Error relativo máximo: {np.max(relative_errors):.2f}%")
    print(f"  - Desviación estándar del error: {np.std(errors):.6f}")
    
    # Calcular R²
    ss_res = np.sum((exact_values - predictions) ** 2)
    ss_tot = np.sum((exact_values - np.mean(exact_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"  - Coeficiente de determinación (R²): {r2:.6f}")
    
    return predictions, exact_values, errors, relative_errors


def plot_results(predictions, exact_values, errors, relative_errors):
    """
    Grafica los resultados de las pruebas.
    
    Args:
        predictions: Predicciones del modelo
        exact_values: Valores exactos
        errors: Errores absolutos
        relative_errors: Errores relativos
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfica 1: Predicciones vs Valores exactos
    axes[0, 0].scatter(exact_values, predictions, alpha=0.6, s=20)
    axes[0, 0].plot([0, max(exact_values)], [0, max(exact_values)], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Valor Exacto')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].set_title('Predicciones vs Valores Exactos')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfica 2: Distribución de errores absolutos
    axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Error Absoluto')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Errores Absolutos')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfica 3: Distribución de errores relativos
    axes[1, 0].hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Error Relativo (%)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Errores Relativos')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfica 4: Error vs Magnitud del número complejo
    magnitudes = np.sqrt(exact_values**2)  # En este caso es lo mismo que exact_values
    axes[1, 1].scatter(magnitudes, errors, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Magnitud del Número Complejo')
    axes[1, 1].set_ylabel('Error Absoluto')
    axes[1, 1].set_title('Error vs Magnitud')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_surface(model, trainer, min_val=-5, max_val=5, resolution=50):
    """
    Grafica la superficie 3D de la función aproximada.
    
    Args:
        model: Modelo entrenado
        trainer: Trainer del modelo
        min_val: Valor mínimo para el rango
        max_val: Valor máximo para el rango
        resolution: Resolución de la malla
    """
    print(f"\n=== Generando Gráfica 3D de la Superficie ===")
    
    # Crear malla
    x = np.linspace(min_val, max_val, resolution)
    y = np.linspace(min_val, max_val, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calcular predicciones
    Z_pred = np.zeros_like(X)
    Z_exact = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            x_re, x_im = X[i, j], Y[i, j]
            prediction, exact = evaluate_model(model, x_re, x_im)
            Z_pred[i, j] = prediction
            Z_exact[i, j] = exact
    
    # Crear figura 3D
    fig = plt.figure(figsize=(15, 6))
    
    # Superficie de predicciones
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x_re')
    ax1.set_ylabel('x_im')
    ax1.set_zlabel('|x| (Predicción)')
    ax1.set_title('Superficie de Predicciones')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Superficie exacta
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x_re')
    ax2.set_ylabel('x_im')
    ax2.set_zlabel('|x| (Exacto)')
    ax2.set_title('Superficie Exacta')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('3d_surface.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Función principal para probar el modelo.
    """
    print("=== Prueba del Modelo de Módulo Complejo ===")
    
    # Cargar modelo entrenado
    model_path = Config().MODEL_PATH
    
    try:
        model, trainer, config = load_trained_model(model_path)
        print(f"Modelo cargado exitosamente desde: {model_path}")
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {model_path}")
        print("Por favor, ejecuta primero el script de entrenamiento (train.py)")
        return
    
    # Probar casos específicos
    test_specific_cases(model, trainer)
    
    # Probar muestras aleatorias
    predictions, exact_values, errors, relative_errors = test_random_samples(model, trainer)
    
    # Graficar resultados
    plot_results(predictions, exact_values, errors, relative_errors)
    
    # Graficar superficie 3D
    plot_3d_surface(model, trainer)
    
    print("\n=== Pruebas Completadas ===")
    print("Gráficas guardadas como 'test_results.png' y '3d_surface.png'")


if __name__ == "__main__":
    main() 