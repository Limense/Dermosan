#  Estructura del Proyecto Dermosan

##  Arquitectura Principal

```
Dermosan/
├── 📄 app.py                          # Aplicación principal Streamlit
├── 📄 README.md                       # Documentación del proyecto
├── 📄 test_system.py                  # Script de pruebas del sistema
├── 📄 verificar_modelo.py             # Script de verificación del modelo
├── 📄 manual_usuario_mejorado.tex     # Manual de usuario en LaTeX
├── 📄 .gitignore                      # Archivos ignorados por Git
│
├── 🗂️ src/                           # Módulos principales
│   ├── 📄 __init__.py                 # Inicialización del paquete
│   ├── 📄 config.py                   # Configuraciones del sistema
│   ├── 📄 predictor.py                # Lógica de predicción IA
│   └── 📄 utils.py                    # Utilidades y componentes UI
│
├── 🗂️ .streamlit/                    # Configuración Streamlit
│   └── 📄 config.toml                 # Configuración de la app
│
├── 🗂️ Modelo Entrenado/              # Modelo de IA
│   └── 📄 best_resnet152.h5           # Modelo ResNet152 entrenado
│
├── 🗂️ Archivos_de_Entrenamiento/     # Archivos de entrenamiento (no en producción)
│   ├── 🗂️ data/                      # Dataset de imágenes dermatológicas
│   └── 🗂️ Codigo_de_entrenamiento/   # Scripts de entrenamiento
│
└── 🗂️ dermosan_venv/                 # Entorno virtual Python (local)
```

##  Archivos Principales

### 📄 **app.py**
- **Propósito:** Interfaz principal de la aplicación web
- **Tecnología:** Streamlit + Plotly
- **Características:** Dashboard médico, análisis de imágenes, reportes

### 📄 **src/predictor.py**
- **Propósito:** Motor de predicción dermatológica
- **Tecnología:** TensorFlow + ResNet152
- **Características:** Análisis de calidad, predicción IA, recomendaciones médicas

### 📄 **src/utils.py**
- **Propósito:** Componentes de UI y utilidades
- **Tecnología:** Streamlit + Plotly
- **Características:** Gráficos médicos, validaciones, exportación

### 📄 **src/config.py**
- **Propósito:** Configuraciones centralizadas
- **Características:** Parámetros del modelo, colores médicos, constantes

##  Archivos de Deployment

###  **Necesarios para Producción:**
- `app.py` - Aplicación principal
- `src/` - Módulos del sistema
- `Modelo Entrenado/best_resnet152.h5` - Modelo IA
- `.streamlit/config.toml` - Configuración
- `requirements.txt` - Dependencias (si existe)

###  **Archivos de Desarrollo:**
- `Archivos_de_Entrenamiento/` - Solo para reentrenamiento
- `test_system.py` - Pruebas del sistema
- `verificar_modelo.py` - Verificación del modelo
- `manual_usuario_mejorado.tex` - Documentación

###  **Excluidos del Repositorio:**
- `dermosan_venv/` - Entorno virtual
- `__pycache__/` - Cache de Python
- `*.log` - Archivos de logs
- Archivos temporales y respaldos

##  Optimizaciones Aplicadas

1. **Eliminados archivos duplicados:** `app_backup.py`, `manual_deusuario.txt`
2. **Removidos módulos no utilizados:** `dashboard_components.py`, `styles.py`
3. **Reorganizada estructura:** Movidos archivos de entrenamiento a carpeta específica
4. **Limpieza de cache:** Eliminados archivos `__pycache__`
5. **Mejorado .gitignore:** Agregadas exclusiones específicas de Dermosan

## 📊 Métricas del Proyecto

- **Archivos principales:** 4 archivos Python
- **Módulos src:** 4 archivos
- **Tamaño modelo:** ~500MB (best_resnet152.h5)
- **Dependencias:** TensorFlow, Streamlit, Plotly, PIL
- **Precisión IA:** 95%

---
*Estructura optimizada - Septiembre 2025*
