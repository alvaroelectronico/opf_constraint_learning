"""
Red neuronal para aproximar el módulo de números complejos.
"""

from .model import ComplexModulusModel
from .trainer import ComplexModulusTrainer, exact_modulus, evaluate_model
from .config import ComplexModulusConfig

__all__ = ['ComplexModulusModel', 'ComplexModulusTrainer', 'ComplexModulusConfig', 'exact_modulus', 'evaluate_model'] 