"""
Script para entrenar la red neuronal del módulo complejo.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from networks.complex_modulus import ComplexModulusModel, ComplexModulusTrainer, ComplexModulusConfig, evaluate_model


def create_data_loaders(config):
    """
    Crea los DataLoaders para entrenamiento y validación.
    
    Args:
        config: Configuración del proyecto
        
    Returns:
        Tuple con (train_loader, val_loader, test_loader)
    """
    # Crear trainer temporal para generar datos
    temp_model = ComplexModulusModel(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    temp_trainer = ComplexModulusTrainer(temp_model, config)
    
    # Generar datos
    inputs, targets = temp_trainer.generate_data()
    
    # Dividir en train, validation y test
    X_temp, X_test, y_temp, y_test = train_test_split(
        inputs, targets, test_size=1-config.TRAIN_SPLIT-config.VALIDATION_SPLIT, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VALIDATION_SPLIT/(config.TRAIN_SPLIT+config.VALIDATION_SPLIT), 
        random_state=42
    )
    
    # Crear datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Datos de entrenamiento: {len(train_dataset)}")
    print(f"Datos de validación: {len(val_dataset)}")
    print(f"Datos de test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def plot_training_history(trainer, save_path="training_history.png"):
    """
    Grafica el historial de entrenamiento.
    
    Args:
        trainer: Trainer con el historial de pérdidas
        save_path: Ruta donde guardar la gráfica
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfica de pérdidas
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss', alpha=0.7)
    plt.plot(trainer.val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.title('Historial de Entrenamiento - Módulo Complejo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfica logarítmica para mejor visualización
    plt.subplot(1, 2, 2)
    plt.semilogy(trainer.train_losses, label='Train Loss', alpha=0.7)
    plt.semilogy(trainer.val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE) - Escala Log')
    plt.title('Historial de Entrenamiento (Escala Log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model_performance(model, test_loader, device):
    """
    Evalúa el rendimiento del modelo en el conjunto de test.
    
    Args:
        model: Modelo entrenado
        test_loader: DataLoader con datos de test
        device: Dispositivo donde ejecutar el modelo
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_inputs)
            loss = torch.nn.MSELoss()(outputs, batch_targets)
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_targets.cpu().numpy().flatten())
    
    # Calcular métricas
    mse = total_loss / len(test_loader)
    rmse = np.sqrt(mse)
    
    # Calcular error relativo promedio
    predictions = np.array(predictions)
    targets = np.array(targets)
    relative_errors = np.abs(predictions - targets) / (targets + 1e-8)
    mean_relative_error = np.mean(relative_errors) * 100
    
    # Calcular R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def main():
    """
    Función principal para entrenar el modelo.
    """
    print("=== Entrenamiento de Red Neuronal para Módulo Complejo ===")
    
    # Configuración
    config = ComplexModulusConfig()
    
    # Crear modelo
    model = ComplexModulusModel(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    print(f"Arquitectura del modelo:")
    print(f"  - Entrada: {config.INPUT_SIZE} (x_re, x_im)")
    print(f"  - Capas ocultas: {config.HIDDEN_SIZES}")
    print(f"  - Salida: {config.OUTPUT_SIZE} (|x|)")
    print(f"  - Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Crear trainer
    trainer = ComplexModulusTrainer(model, config)
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Entrenar modelo
    print("\n=== Iniciando Entrenamiento ===")
    history = trainer.train(train_loader, val_loader)
    
    # Guardar modelo
    trainer.save_model()
    
    # Evaluar rendimiento
    print("\n=== Evaluando Rendimiento ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    performance = evaluate_model_performance(model, test_loader, device)
    
    print(f"Métricas de rendimiento en test:")
    print(f"  - MSE: {performance['mse']:.6f}")
    print(f"  - RMSE: {performance['rmse']:.6f}")
    print(f"  - Error relativo promedio: {performance['mean_relative_error']:.2f}%")
    print(f"  - R²: {performance['r2']:.6f}")
    
    # Graficar historial de entrenamiento
    plot_training_history(trainer)
    
    # Ejemplos de predicciones
    print("\n=== Ejemplos de Predicciones ===")
    test_cases = [
        (1.0, 0.0),    # |1| = 1
        (0.0, 1.0),    # |i| = 1
        (3.0, 4.0),    # |3+4i| = 5
        (-5.0, 12.0),  # |-5+12i| = 13
        (0.5, 0.5),    # |0.5+0.5i| = 0.707...
    ]
    
    for x_re, x_im in test_cases:
        prediction, exact = evaluate_model(model, x_re, x_im)
        error = abs(prediction - exact)
        print(f"|{x_re}+{x_im}i|: Predicción={prediction:.6f}, Exacto={exact:.6f}, Error={error:.6f}")
    
    print("\n=== Entrenamiento Completado ===")


if __name__ == "__main__":
    main() 