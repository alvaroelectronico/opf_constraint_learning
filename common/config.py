"""
Configuración base para todas las redes neuronales del proyecto OPF.
"""

class BaseConfig:
    """
    Clase base de configuración que contiene parámetros comunes.
    """
    
    # Parámetros de entrenamiento comunes
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    
    # Parámetros de generación de datos comunes
    NUM_SAMPLES = 10000
    MIN_VALUE = -10.0
    MAX_VALUE = 10.0
    
    # Parámetros de visualización
    PLOT_SAMPLES = 1000
    
    # Directorio base para modelos
    MODELS_BASE_PATH = "models"
    
    def __init__(self, network_name: str):
        """
        Inicializa la configuración base.
        
        Args:
            network_name: Nombre de la red neuronal
        """
        self.network_name = network_name
        self.MODEL_PATH = f"{self.MODELS_BASE_PATH}/{network_name}/model.pth"
    
    def get_model_dir(self) -> str:
        """
        Obtiene el directorio específico para el modelo.
        
        Returns:
            Ruta del directorio del modelo
        """
        return f"{self.MODELS_BASE_PATH}/{self.network_name}"
    
    def to_dict(self) -> dict:
        """
        Convierte la configuración a diccionario.
        
        Returns:
            Diccionario con la configuración
        """
        return {
            'network_name': self.network_name,
            'learning_rate': self.LEARNING_RATE,
            'batch_size': self.BATCH_SIZE,
            'num_epochs': self.NUM_EPOCHS,
            'train_split': self.TRAIN_SPLIT,
            'validation_split': self.VALIDATION_SPLIT,
            'num_samples': self.NUM_SAMPLES,
            'min_value': self.MIN_VALUE,
            'max_value': self.MAX_VALUE,
            'model_path': self.MODEL_PATH
        } 