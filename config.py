"""
Configuración para la red neuronal que aproxima el módulo de números complejos.
"""

class Config:
    # Parámetros de la red neuronal
    INPUT_SIZE = 2  # x_re, x_im
    HIDDEN_SIZES = [64, 32, 16]  # Capas ocultas
    OUTPUT_SIZE = 1  # |x|
    
    # Parámetros de entrenamiento
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    
    # Generación de datos
    NUM_SAMPLES = 10000
    MIN_VALUE = -10.0
    MAX_VALUE = 10.0
    
    # Guardado de modelo
    MODEL_PATH = "models/complex_modulus_model.pth"
    
    # Parámetros de visualización
    PLOT_SAMPLES = 1000 