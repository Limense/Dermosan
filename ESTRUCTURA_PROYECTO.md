# 📁 Estructura del Proyecto DERMOSAN

## 🏗️ Arquitectura Principal

```
dermosan/
├── 📄 app.py                         # Aplicación principal Streamlit
├── 📄 README.md                      # Documentación del proyecto
├── 📄 test_system.py                 # Script de pruebas del sistema
├── 📄 verificar_modelo.py            # Script de verificación del modelo
├── 📄 .gitignore                     # Archivos ignorados por Git
│
├── 🗂️ src/                           # Módulos principales
│   ├── 📄 __init__.py                # Inicialización del paquete
│   ├── 📄 config.py                  # Configuraciones del sistema
│   ├── 📄 predictor.py               # Lógica de predicción IA
│   └── 📄 utils.py                   # Utilidades y componentes UI
│
├── 🗂️ Modelo Entrenado/              # Acá irá el modelo entrenado (.h5) después del entrenamiento
│   └── 📄 best_resnet152.h5          # Modelo final ResNet152 (generado tras el entrenamiento)
│
└── 🗂️ Codigo Entrenamiento/          # Scripts para entrenamiento
    ├── 📄 train_resnet152.py         # Entrenamiento del modelo de IA
    └── 🗂️ IMG_CLASSES/               # Acá irá el dataset de Kaggle: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset 

```

##  Archivos Principales

### 📄 **app.py**
- **Propósito:** Interfaz principal de la aplicación web  
- **Tecnología:** Streamlit + Plotly  
- **Características:** Dashboard médico, análisis de imágenes, generación de reportes

### 📄 **src/predictor.py**
- **Propósito:** Motor de predicción dermatológica  
- **Tecnología:** TensorFlow + ResNet152  
- **Características:** Análisis de calidad, predicción IA, recomendaciones médicas

### 📄 **src/utils.py**
- **Propósito:** Componentes de interfaz y utilidades  
- **Tecnología:** Streamlit + Plotly  
- **Características:** Gráficos médicos, validaciones, exportación de resultados

### 📄 **src/config.py**
- **Propósito:** Configuraciones centralizadas  
- **Características:** Parámetros del modelo, umbrales de confianza, constantes globales

---

##  Archivos de Deployment

### ✅ **Necesarios para Producción:**
- `app.py` – Aplicación principal  
- `src/` – Módulos del sistema  
- `Modelo Entrenado/best_resnet152.h5` – Modelo de IA entrenado  
- `requirements.txt` – Dependencias del entorno  

### 🧩 **Archivos de Desarrollo:**
- `Codigo Entrenamiento/` – Scripts para entrenamiento o reentrenamiento
- `IMG_CLASSES/` – Dataset de Kaggle para el entrenamiento  
- `test_system.py` – Pruebas funcionales del sistema  
- `verificar_modelo.py` – Validación del modelo entrenado  

### 🚫 **Excluidos del Repositorio:**
- `dermosan_venv/` – Entorno virtual local  
- `__pycache__/` – Caché de Python
- `*.log` – Archivos de logs  
- Archivos temporales y respaldos automáticos  


---

##  Optimizaciones Aplicadas

1. **✅ Reorganizada la estructura:** Archivos de entrenamiento movidos a `Codigo Entrenamiento/`  
2. **✅ Eliminados archivos obsoletos y duplicados**  
3. **✅ Limpieza de caché:** Removidos archivos `__pycache__`  
4. **✅ Actualizado `.gitignore`:** Se agregaron exclusiones específicas del proyecto  
5. **✅ Estandarizada la nomenclatura de carpetas y archivos**

---

## 📊 Métricas del Proyecto

- **Archivos principales:** 4 archivos Python  
- **Módulos en `src/`:** 4 archivos  
- **Tamaño del modelo:** ~500 MB (`best_resnet152.h5`)  
- **Dependencias clave:** TensorFlow, Streamlit, Plotly, Pillow  
- **Precisión estimada del modelo:** ~95%  

---

*Estructura optimizada – Octubre 2025*
