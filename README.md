# Dermosan - Sistema de Diagnóstico Dermatológico

Herramienta Interactiva con Streamlit para el Diagnóstico Dermatológico Asistido mediante Redes Neuronales Convolucionales (CNNs)

## 🏥 Descripción del Proyecto

Repositorio oficial del proyecto DERMOSAN, una herramienta de diagnóstico dermatológico asistido por inteligencia artificial. Integra un modelo ResNet152 entrenado sobre 27 153 imágenes dermatológicas y una interfaz Streamlit para la carga, análisis y visualización de resultados en tiempo real. 

### 🎯 Enfermedades Diagnosticadas

1. **Eczema** *(1,677 imágenes)*
2. **Melanoma** *(3,140 imágenes)*
3. **Atopic Dermatitis** *(1,257 imágenes)*
4. **Basal Cell Carcinoma (BCC)** *(3,323 imágenes)*
5. **Melanocytic Nevi (NV)** *(7,970 imágenes)*
6. **Benign Keratosis-like Lesions (BKL)** *(2,624 imágenes)*
7. **Psoriasis / Lichen Planus & related diseases** *(2,055 imágenes)*
8. **Seborrheic Keratoses & other benign tumors** *(1,847 imágenes)*
9. **Tinea / Candidiasis & other fungal infections** *(1,702 imágenes)*
10. **Warts / Molluscum & other viral infections** *(2,103 imágenes)*

**Total general:** *27,153 imágenes*



## 🚀 Características Principales

- **Modelo de Deep Learning:** Basado en arquitectura **ResNet152** con *transfer learning*  
- **Interfaz interactiva:** Desarrollada con **Streamlit** para diagnóstico en tiempo real  
- **Análisis de calidad de imagen:** Evaluación automática de nitidez, iluminación y resolución  
- **Diagnóstico diferencial:** Visualización de las **Top 3 predicciones** con probabilidades  
- **Recomendaciones clínicas:** Sugerencias médicas interpretativas según nivel de confianza  
- **Reportes exportables:** Generación automática de informes de diagnóstico en formato texto  
- **Validación médica:** Recordatorio explícito de interpretación por un profesional de salud



### 🧩 Dependencias principales

- **Python 3.10+**
- **TensorFlow 2.15.0** y **Keras 3.0**
- **Streamlit 1.32.0**
- **NumPy**, **Pandas**, **Plotly**
- **OpenCV**, **Pillow**
- **Scikit-learn**
- **Matplotlib** y **Seaborn**


## 🛠️ Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tavoofg/dermosan.git
cd dermosan
```

2. **Crear entorno virtual:**
```bash
python -m venv dermosan_env
# Windows
dermosan_env\Scripts\activate
# Linux/Mac
source dermosan_env/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Verificar estructura de archivos:**
```
dermosan/
├── app.py                  # Aplicación principal (interfaz Streamlit)
├── requirements.txt        # Dependencias del entorno
├── src/                    # Módulos del sistema
│ ├── config.py                 # Configuración general del sistema
│ ├── predictor.py              # Módulo de predicción
│ └── utils.py                  # Utilidades de interfaz
├── Modelo Entrenado/       # 📁 Acá irá el modelo entrenado (.h5)
│ └── best_resnet152.h5         # Modelo final ResNet152
├── Codigo Entrenamiento/   # Scripts para entrenamiento
│ └── train_resnet152.py        # Entrenamiento del modelo      
└── verificar_modelo.py     # Validación del modelo
```

## 🏃‍♂️ Uso del Sistema

### Ejecutar la aplicación:
```bash
streamlit run app.py
```

### Acceder a la interfaz:
- Abrir navegador en: `http://localhost:8501`
- Subir imagen dermatológica (JPG, PNG)
- Revisar análisis de calidad
- Obtener diagnóstico y recomendaciones
- Generar reporte médico

## 📊 Rendimiento del Modelo

- **Arquitectura**: ResNet152 con *transfer learning*  
- **Precisión estimada**: ~95%  
- **Dataset**: 27,000+ imágenes dermatológicas  
- **Validación**: División 80/10/10 (train/val/test)  
- **Optimizaciones**: *Data augmentation*, *class balancing*  

## 🔧 Estructura del Código

### `app.py`
Aplicación principal de **Streamlit** con:
- Interfaz de usuario completa  
- Gestión de carga de imágenes  
- Visualización de resultados  
- Generación de reportes exportables  

### `src/predictor.py`
Módulo de predicción que incluye:
- Carga y gestión del modelo entrenado  
- Preprocesamiento de imágenes dermatológicas  
- Generación de predicciones  
- Análisis automático de calidad de imagen  

### `src/config.py`
Configuración centralizada:
- Parámetros del modelo  
- Información médica de las enfermedades  
- Umbrales de confianza  
- Configuración general de la aplicación  

### `src/utils.py`
Utilidades de interfaz:
- Componentes visuales de **Streamlit**  
- Gráficos interactivos y visualizaciones  
- Formateo de resultados clínicos  
- Exportación de reportes médicos  

## 🏥 Uso Clínico

### Recomendaciones:
1. **Uso como herramienta de apoyo** – No reemplaza el criterio médico profesional.  
2. **Validación profesional** – Confirmar siempre con un dermatólogo certificado.  
3. **Calidad de imagen** – Utilizar fotografías nítidas y bien iluminadas.  
4. **Casos urgentes** – Atención inmediata para melanomas o lesiones sospechosas.  

### Limitaciones:
- No diagnostica todas las condiciones dermatológicas.  
- Requiere validación por un profesional médico.  
- Su rendimiento depende de la calidad de la imagen.  
- No reemplaza la biopsia ni otros estudios confirmatorios.  

## 🔬 Entrenamiento del Modelo

El modelo fue entrenado usando:
- **Base**: ResNet152 preentrenado en *ImageNet*  
- **Fine-tuning**: Últimas 50 capas entrenables  
- **Optimizador**: Adam con *learning rate* adaptativo  
- **Augmentación**: Rotación, zoom, volteo horizontal  
- **Balanceamiento**: Pesos de clase automáticos para dataset desbalanceado  

## 📈 Mejoras Futuras

- [ ] Versión móvil para dispositivos médicos  
- [ ] Inclusión de más clases dermatológicas  
- [ ] Análisis simultáneo de múltiples lesiones  
- [ ] Seguimiento temporal de casos clínicos  
- [ ] API REST para integración con otros sistemas


## 👥 Equipo de Desarrollo

Desarrollado en el marco del proyecto de investigación **“DERMOSAN – Sistema de Diagnóstico Dermatológico Asistido por Inteligencia Artificial”**, como parte de las iniciativas de innovación promovidas por la **Universidad Nacional de Cañete (UNDC)**.

**Integrantes del equipo:**
- **Gustavo Fernández-Gutiérrez** – Desarrollador principal e investigador responsable  
- **Andry Diego-Calagua** – Desarrollador e investigador asociado  
- **Alex Pacheco-Pumaleque** – Asesor académico  

**Entidad financiadora:** Dirección de Innovación y Transferencia Tecnológica (DITT) – UNDC



## 📄 Licencia

Este proyecto se distribuye bajo la licencia [MIT](https://opensource.org/licenses/MIT).  
**DERMOSAN – UNDC 2025** fue desarrollado con fines de investigación y aplicación médica, orientado al apoyo diagnóstico dermatológico mediante inteligencia artificial.



## 💰 Información sobre Subvención

Este trabajo fue financiado por la **Dirección de Innovación y Transferencia Tecnológica (DITT)** de la **Vicepresidencia de Investigación** de la **Universidad Nacional de Cañete (UNDC)**, en el marco del **“Concurso de Investigación para el Desarrollo de Innovaciones y Propiedad Intelectual”**, bajo el **número de contrato 017-2024**.


## ⚠️ Disclaimer Médico

Este sistema es una **herramienta de apoyo al diagnóstico** y **no reemplaza**:

- El criterio clínico profesional  
- La evaluación médica presencial  
- Los estudios complementarios necesarios  
- La biopsia cuando esté indicada  

Siempre consulte con un **dermatólogo certificado** para la confirmación diagnóstica y el tratamiento adecuado.

---

**Dermosan v1.0.0** – Sistema de Diagnóstico Dermatológico Automatizado  
*DERMOSAN – UNDC 2025*