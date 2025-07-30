"""
Script para mostrar información detallada de modelos guardados.
"""

import torch
import os
from config import Config


def load_model_info(model_path: str):
    """
    Carga y muestra información detallada de un modelo guardado.
    
    Args:
        model_path: Ruta del modelo a analizar
    """
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


def print_model_summary(checkpoint):
    """
    Imprime un resumen completo del modelo.
    
    Args:
        checkpoint: Checkpoint cargado del modelo
    """
    print("=" * 60)
    print("INFORMACIÓN DEL MODELO GUARDADO")
    print("=" * 60)
    
    # Información de la arquitectura
    if 'model_info' in checkpoint:
        model_info = checkpoint['model_info']
        print("\n📊 ARQUITECTURA DE LA RED:")
        print(f"  • Entrada: {model_info['input_size']} neuronas (x_re, x_im)")
        print(f"  • Capas ocultas: {model_info['hidden_sizes']}")
        print(f"  • Salida: {model_info['output_size']} neurona (|x|)")
        print(f"  • Parámetros totales: {model_info['total_parameters']:,}")
        print(f"  • Parámetros entrenables: {model_info['trainable_parameters']:,}")
    
    # Información de entrenamiento
    if 'training_info' in checkpoint:
        training_info = checkpoint['training_info']
        print("\n🎯 CONFIGURACIÓN DE ENTRENAMIENTO:")
        print(f"  • Learning rate: {training_info['learning_rate']}")
        print(f"  • Batch size: {training_info['batch_size']}")
        print(f"  • Épocas configuradas: {training_info['num_epochs']}")
        print(f"  • Split train/val: {training_info['train_split']:.1%}/{training_info['validation_split']:.1%}")
        
        if training_info['final_train_loss'] is not None:
            print(f"  • Pérdida final (train): {training_info['final_train_loss']:.6f}")
        if training_info['final_val_loss'] is not None:
            print(f"  • Pérdida final (val): {training_info['final_val_loss']:.6f}")
        if training_info['best_val_loss'] is not None:
            print(f"  • Mejor pérdida (val): {training_info['best_val_loss']:.6f}")
        if training_info['best_epoch'] is not None:
            print(f"  • Mejor época: {training_info['best_epoch'] + 1}")
    
    # Información de datos
    if 'data_info' in checkpoint:
        data_info = checkpoint['data_info']
        print("\n📈 CONFIGURACIÓN DE DATOS:")
        print(f"  • Muestras totales: {data_info['num_samples']:,}")
        print(f"  • Rango de valores: {data_info['data_range']}")
        print(f"  • Valor mínimo: {data_info['min_value']}")
        print(f"  • Valor máximo: {data_info['max_value']}")
    
    # Información del dispositivo
    if 'device_info' in checkpoint:
        device_info = checkpoint['device_info']
        print("\n💻 CONFIGURACIÓN TÉCNICA:")
        print(f"  • Dispositivo usado: {device_info['device_used']}")
        print(f"  • Optimizador: {device_info['optimizer_type']}")
        print(f"  • Función de pérdida: {device_info['criterion_type']}")
    
    # Historial de entrenamiento
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        if train_losses and val_losses:
            print("\n📉 HISTORIAL DE ENTRENAMIENTO:")
            print(f"  • Épocas entrenadas: {len(train_losses)}")
            print(f"  • Pérdida inicial (train): {train_losses[0]:.6f}")
            print(f"  • Pérdida inicial (val): {val_losses[0]:.6f}")
            print(f"  • Pérdida final (train): {train_losses[-1]:.6f}")
            print(f"  • Pérdida final (val): {val_losses[-1]:.6f}")
            print(f"  • Mejora (train): {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
            print(f"  • Mejora (val): {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.2f}%")
    
    # Timestamp
    if 'save_timestamp' in checkpoint and checkpoint['save_timestamp']:
        print(f"\n⏰ Guardado el: {checkpoint['save_timestamp']}")
    
    print("\n" + "=" * 60)


def compare_models(model_paths):
    """
    Compara múltiples modelos guardados.
    
    Args:
        model_paths: Lista de rutas de modelos a comparar
    """
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    models_data = []
    
    for path in model_paths:
        if os.path.exists(path):
            checkpoint = load_model_info(path)
            if checkpoint:
                models_data.append((path, checkpoint))
    
    if not models_data:
        print("No se encontraron modelos válidos para comparar.")
        return
    
    # Comparar arquitecturas
    print("\n🏗️  COMPARACIÓN DE ARQUITECTURAS:")
    print(f"{'Modelo':<30} {'Parámetros':<15} {'Capas Ocultas':<20}")
    print("-" * 65)
    
    for path, checkpoint in models_data:
        model_name = os.path.basename(path)
        if 'model_info' in checkpoint:
            info = checkpoint['model_info']
            params = f"{info['total_parameters']:,}"
            layers = str(info['hidden_sizes'])
            print(f"{model_name:<30} {params:<15} {layers:<20}")
    
    # Comparar rendimiento
    print("\n📊 COMPARACIÓN DE RENDIMIENTO:")
    print(f"{'Modelo':<30} {'Mejor Val Loss':<15} {'Épocas':<10}")
    print("-" * 55)
    
    for path, checkpoint in models_data:
        model_name = os.path.basename(path)
        if 'training_info' in checkpoint:
            info = checkpoint['training_info']
            best_loss = f"{info['best_val_loss']:.6f}" if info['best_val_loss'] else "N/A"
            epochs = len(checkpoint.get('train_losses', []))
            print(f"{model_name:<30} {best_loss:<15} {epochs:<10}")


def main():
    """
    Función principal para mostrar información de modelos.
    """
    config = Config()
    model_path = config.MODEL_PATH
    
    print("🔍 ANALIZADOR DE MODELOS GUARDADOS")
    print("=" * 60)
    
    # Mostrar información del modelo principal
    if os.path.exists(model_path):
        print(f"\n📁 Modelo encontrado: {model_path}")
        checkpoint = load_model_info(model_path)
        if checkpoint:
            print_model_summary(checkpoint)
        else:
            print("❌ Error al cargar el modelo.")
    else:
        print(f"❌ No se encontró el modelo en: {model_path}")
        print("💡 Ejecuta 'python train.py' para entrenar un modelo.")
    
    # Buscar otros modelos en el directorio
    models_dir = os.path.dirname(model_path)
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if len(model_files) > 1:
            print(f"\n🔍 Se encontraron {len(model_files)} modelos en {models_dir}:")
            for i, model_file in enumerate(model_files, 1):
                print(f"  {i}. {model_file}")
            
            # Comparar modelos si hay más de uno
            if len(model_files) > 1:
                model_paths = [os.path.join(models_dir, f) for f in model_files]
                compare_models(model_paths)


if __name__ == "__main__":
    main() 