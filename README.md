# opf_constraint_learning

Este proyecto tiene por objetivo final construir un modelo basado en constraint learning para el problema OPF.

El primer paso es modelizar las relaciones no lineales del problema mediante redes neuronales.

Por el momento, solo quiero crear una red neuronal que aproxime el módulo de un número complejo, es decir, que reciba como argumento x_re y x_im (partes real e imaginaria, respectivamente) y que aproxime (x_re**2 + x_im**2)**0.5

## Estructura del Proyecto

```
opf_constraint_learning/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias del proyecto
├── config.py                # Configuración y hiperparámetros
├── complex_modulus_nn.py    # Red neuronal y trainer
├── train.py                 # Script de entrenamiento
├── test.py                  # Script de pruebas y evaluación
├── example.py               # Ejemplo de uso simple
├── model_info.py            # Analizador de modelos guardados
└── models/                  # Directorio para guardar modelos (se crea automáticamente)
```

## Instalación

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejemplo Rápido
```bash
python example.py
```

### Entrenamiento
```bash
python train.py
```

### Pruebas
```bash
python test.py
```

### Información del Modelo
```bash
python model_info.py
```

## Arquitectura de la Red Neuronal

La red neuronal implementada tiene la siguiente estructura:
- **Entrada**: 2 neuronas (x_re, x_im)
- **Capas ocultas**: [64, 32, 16] neuronas con ReLU y Dropout
- **Salida**: 1 neurona (|x|)

## Características

- ✅ Entrenamiento con datos sintéticos generados automáticamente
- ✅ Validación y métricas de rendimiento
- ✅ Visualización de resultados y superficies 3D
- ✅ Guardado y carga de modelos entrenados
- ✅ Información detallada de configuración guardada
- ✅ Analizador de modelos con comparaciones
- ✅ Soporte para GPU/CPU automático
- ✅ Documentación completa en español