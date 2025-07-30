# opf_constraint_learning

Este proyecto tiene por objetivo final construir un modelo basado en constraint learning para el problema OPF.

El primer paso es modelizar las relaciones no lineales del problema mediante redes neuronales.

Por el momento, solo quiero crear una red neuronal que aproxime el módulo de un número complejo, es decir, que reciba como argumento x_re y x_im (partes real e imaginaria, respectivamente) y que aproxime (x_re**2 + x_im**2)**0.5

## Estructura del Proyecto

```
opf_constraint_learning/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias del proyecto
├── common/                   # Código común reutilizable
│   ├── __init__.py
│   ├── base_trainer.py       # Clase base para trainers
│   ├── base_model.py         # Clase base para modelos
│   ├── config.py             # Configuración base
│   └── model_info.py         # Analizador de modelos
├── networks/                 # Redes neuronales específicas
│   ├── __init__.py
│   ├── complex_modulus/      # Red para módulo complejo
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── config.py
│   │   └── train.py
│   └── power_flow/           # Red para flujo de potencia
│       ├── __init__.py
│       ├── model.py
│       ├── trainer.py
│       ├── config.py
│       └── train.py
├── models/                   # Modelos entrenados
│   ├── complex_modulus/
│   └── power_flow/
└── scripts/                  # Scripts de utilidad
    └── __init__.py
```

## Instalación

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejemplo Rápido
```bash
python scripts/example_usage.py
```

### Entrenamiento

#### Red de Módulo Complejo
```bash
python networks/complex_modulus/train.py
```

#### Red de Flujo de Potencia
```bash
python networks/power_flow/train.py
```

### Información del Modelo
```bash
python common/model_info.py
```

## Redes Neuronales Implementadas

### 1. Red de Módulo Complejo
- **Entrada**: 2 neuronas (x_re, x_im)
- **Capas ocultas**: [64, 32, 16] neuronas con ReLU y Dropout
- **Salida**: 1 neurona (|x|)
- **Función**: Aproxima |x| = sqrt(x_re² + x_im²)

### 2. Red de Flujo de Potencia
- **Entrada**: 6 neuronas (u_Ri, u_Ii, u_Rj, u_Ij, G_ij, B_ij)
- **Capas ocultas**: [128, 64, 32] neuronas con ReLU y Dropout
- **Salida**: 2 neuronas (p_ij, q_ij)
- **Función**: Aproxima las ecuaciones de flujo de potencia:
  - p_ij = G_ij × (u_Ri² + u_Ii² - u_Ri×u_Rj - u_Ii×u_Ij) + B_ij × (u_Ri×u_Ij - u_Ii×u_Rj)
  - q_ij = -B_ij × (u_Ri² + u_Ii² - u_Ri×u_Rj - u_Ii×u_Ij) + G_ij × (u_Ri×u_Ij - u_Ii×u_Rj)

## Características

- ✅ Entrenamiento con datos sintéticos generados automáticamente
- ✅ Validación y métricas de rendimiento
- ✅ Visualización de resultados y superficies 3D
- ✅ Guardado y carga de modelos entrenados
- ✅ Información detallada de configuración guardada
- ✅ Analizador de modelos con comparaciones
- ✅ Soporte para GPU/CPU automático
- ✅ Documentación completa en español