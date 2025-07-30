"""
Script para entrenar la red neuronal de flujo de potencia.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from networks.power_flow import PowerFlowModel, PowerFlowTrainer, PowerFlowConfig, evaluate_model


def create_data_loaders(config):
    """
    Crea los DataLoaders para entrenamiento y validación.
    
    Args:
        config: Configuración del proyecto
        
    Returns:
        Tuple con (train_loader, val_loader, test_loader)
    """
    # Crear trainer temporal para generar datos
    temp_model = PowerFlowModel(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    temp_trainer = PowerFlowTrainer(temp_model, config)
    
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


def plot_training_history(trainer, save_path="training_history_power_flow.png"):
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
    plt.title('Historial de Entrenamiento - Flujo de Potencia')
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
    predictions_p = []
    predictions_q = []
    targets_p = []
    targets_q = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_inputs)
            loss = torch.nn.MSELoss()(outputs, batch_targets)
            total_loss += loss.item()
            
            # Separar predicciones de P y Q
            pred_p = outputs[:, 0].cpu().numpy()
            pred_q = outputs[:, 1].cpu().numpy()
            target_p = batch_targets[:, 0].cpu().numpy()
            target_q = batch_targets[:, 1].cpu().numpy()
            
            predictions_p.extend(pred_p)
            predictions_q.extend(pred_q)
            targets_p.extend(target_p)
            targets_q.extend(target_q)
    
    # Calcular métricas
    mse = total_loss / len(test_loader)
    rmse = np.sqrt(mse)
    
    # Calcular errores para P y Q por separado
    predictions_p = np.array(predictions_p)
    predictions_q = np.array(predictions_q)
    targets_p = np.array(targets_p)
    targets_q = np.array(targets_q)
    
    # Error relativo para P
    relative_errors_p = np.abs(predictions_p - targets_p) / (np.abs(targets_p) + 1e-8)
    mean_relative_error_p = np.mean(relative_errors_p) * 100
    
    # Error relativo para Q
    relative_errors_q = np.abs(predictions_q - targets_q) / (np.abs(targets_q) + 1e-8)
    mean_relative_error_q = np.mean(relative_errors_q) * 100
    
    # R² para P y Q
    ss_res_p = np.sum((targets_p - predictions_p) ** 2)
    ss_tot_p = np.sum((targets_p - np.mean(targets_p)) ** 2)
    r2_p = 1 - (ss_res_p / ss_tot_p)
    
    ss_res_q = np.sum((targets_q - predictions_q) ** 2)
    ss_tot_q = np.sum((targets_q - np.mean(targets_q)) ** 2)
    r2_q = 1 - (ss_res_q / ss_tot_q)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mean_relative_error_p': mean_relative_error_p,
        'mean_relative_error_q': mean_relative_error_q,
        'r2_p': r2_p,
        'r2_q': r2_q,
        'predictions_p': predictions_p,
        'predictions_q': predictions_q,
        'targets_p': targets_p,
        'targets_q': targets_q
    }


def main():
    """
    Función principal para entrenar el modelo.
    """
    print("=== Entrenamiento de Red Neuronal para Flujo de Potencia ===")
    
    # Configuración
    config = PowerFlowConfig()
    
    # Crear modelo
    model = PowerFlowModel(
        input_size=config.INPUT_SIZE,
        hidden_sizes=config.HIDDEN_SIZES,
        output_size=config.OUTPUT_SIZE
    )
    
    print(f"Arquitectura del modelo:")
    print(f"  - Entrada: {config.INPUT_SIZE} (u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij)")
    print(f"  - Capas ocultas: {config.HIDDEN_SIZES}")
    print(f"  - Salida: {config.OUTPUT_SIZE} (p_ij, q_ij)")
    print(f"  - Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Crear trainer
    trainer = PowerFlowTrainer(model, config)
    
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
    print(f"  - Error relativo promedio P: {performance['mean_relative_error_p']:.2f}%")
    print(f"  - Error relativo promedio Q: {performance['mean_relative_error_q']:.2f}%")
    print(f"  - R² para P: {performance['r2_p']:.6f}")
    print(f"  - R² para Q: {performance['r2_q']:.6f}")
    
    # Graficar historial de entrenamiento
    plot_training_history(trainer)
    
    # Ejemplos de predicciones
    print("\n=== Ejemplos de Predicciones ===")
    test_cases = [
        (1.0, 0.0, 0.95, 0.0, 1.0, -2.0),    # Caso típico
        (1.0, 0.1, 0.9, 0.05, 0.5, -1.5),    # Con componentes imaginarias
        (0.95, 0.0, 1.0, 0.0, 2.0, -3.0),    # Alta conductancia
        (1.0, 0.0, 0.8, 0.0, 0.1, -0.5),     # Baja conductancia
    ]
    
    for u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij in test_cases:
        (pred_p, pred_q), (exact_p, exact_q) = evaluate_model(model, u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij)
        error_p = abs(pred_p - exact_p)
        error_q = abs(pred_q - exact_q)
        print(f"P: Pred={pred_p:.6f}, Exact={exact_p:.6f}, Error={error_p:.6f}")
        print(f"Q: Pred={pred_q:.6f}, Exact={exact_q:.6f}, Error={error_q:.6f}")
        print("-" * 50)
    
    print("\n=== Entrenamiento Completado ===")


if __name__ == "__main__":
    main() 