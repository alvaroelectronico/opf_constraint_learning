"""
Módulo común para todas las redes neuronales del proyecto OPF.
"""

from .config import BaseConfig
from .base_model import BaseModel
from .base_trainer import BaseTrainer

__all__ = ['BaseConfig', 'BaseModel', 'BaseTrainer'] 