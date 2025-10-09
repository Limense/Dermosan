# Dermosan - Sistema de Diagnóstico Dermatológico

Herramienta Interactiva con Streamlit para el Diagnóstico Dermatológico Asistido mediante Redes Neuronales Convolucionales (CNNs)

##  Descripción del Proyecto

Dermosan es un sistema de inteligencia artificial diseñado para asistir a los profesionales médicos en el diagnóstico de 10 tipos diferentes de enfermedades dermatológicas. Utiliza un modelo ResNet152 entrenado con más de 30,000 imágenes dermatológicas.

###  Enfermedades Diagnosticadas

1. **Eczema** (1,677 imágenes)
2. **Warts Molluscum and Viral Infections** (2,103 imágenes)
3. **Melanoma** (15,750 imágenes)
4. **Atopic Dermatitis** (1,250 imágenes)
5. **Basal Cell Carcinoma (BCC)** (3,323 imágenes)
6. **Melanocytic Nevi (NV)** (7,970 imágenes)
7. **Benign Keratosis-like Lesions (BKL)** (2,624 imágenes)
8. **Psoriasis Lichen Planus** (2,000 imágenes)
9. **Seborrheic Keratoses** (1,800 imágenes)
10. **Tinea Ringworm Candidiasis** (1,700 imágenes)

##  Características Principales

- **Modelo de Deep Learning**: ResNet152 con transfer learning
- **Interfaz intuitiva**: Desarrollada con Streamlit
- **Análisis de calidad de imagen**: Evaluación automática de idoneidad
- **Diagnóstico diferencial**: Top 3 predicciones con probabilidades
- **Recomendaciones clínicas**: Sugerencias médicas basadas en confianza
- **Reportes exportables**: Generación de informes completos
- **Validación médica**: Recordatorios de validación profesional

##  Requisitos del Sistema

### Dependencias principales:
- Python 3.8+
- TensorFlow 2.15.0
- Streamlit 1.32.0
- NumPy, Pandas, Plotly
- OpenCV, Pillow
- Scikit-learn

##  Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/Limense/Dermosan.git
cd Dermosan
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
Dermosan/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias
├── src/
│   ├── config.py            # Configuración del sistema
│   ├── predictor.py         # Módulo de predicción
│   └── utils.py             # Utilidades de interfaz
├── Modelo Entrenado/
│   └── best_resnet152.h5    # Modelo entrenado
├── Codigo de entrenamiento/
│   └── script.py            # Script de entrenamiento
└── Datos de prueba/         # Imágenes para testing
```

##  Uso del Sistema

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

##  Rendimiento del Modelo

- **Arquitectura**: ResNet152 con transfer learning
- **Precisión estimada**: ~95%
- **Dataset**: 30,000+ imágenes dermatológicas
- **Validación**: Split 80/10/10 (train/val/test)
- **Optimizaciones**: Data augmentation, class balancing

##  Estructura del Código

### `app.py`
Aplicación principal de Streamlit con:
- Interfaz de usuario completa
- Gestión de uploads de imágenes
- Visualización de resultados
- Generación de reportes

### `src/predictor.py`
Módulo de predicción que incluye:
- Carga y gestión del modelo
- Preprocesamiento de imágenes
- Generación de predicciones
- Análisis de calidad de imagen

### `src/config.py`
Configuración centralizada:
- Parámetros del modelo
- Información de enfermedades
- Thresholds de confianza
- Configuración de la app

### `src/utils.py`
Utilidades de interfaz:
- Componentes de Streamlit
- Gráficos y visualizaciones
- Formateo de resultados
- Exportación de reportes

##  Uso Clínico

### Recomendaciones:
1. **Uso como herramienta de apoyo** - No reemplaza criterio médico
2. **Validación profesional** - Siempre confirmar con dermatólogo
3. **Calidad de imagen** - Usar fotografías nítidas y bien iluminadas
4. **Casos urgentes** - Atención inmediata para melanomas sospechosos

### Limitaciones:
- No diagnostica todas las condiciones dermatológicas
- Requiere validación por profesional médico
- Dependiente de calidad de imagen
- No reemplaza biopsia cuando sea necesaria

##  Entrenamiento del Modelo

El modelo fue entrenado usando:
- **Base**: ResNet152 pre-entrenado en ImageNet
- **Fine-tuning**: Últimas 50 capas entrenables
- **Optimizador**: Adam con learning rate adaptativo
- **Augmentación**: Rotación, zoom, flip horizontal
- **Balanceamiento**: Pesos de clase para dataset desbalanceado

##  Mejoras Futuras

- [ ] Integración con sistemas hospitalarios (HL7 FHIR)
- [ ] Versión móvil para dispositivos médicos
- [ ] Más clases de enfermedades dermatológicas
- [ ] Análisis de múltiples lesiones
- [ ] Seguimiento temporal de lesiones
- [ ] API REST para integración

##  Equipo de Desarrollo

Desarrollado en el marco del proyecto de investigación sobre diagnóstico dermatológico automatizado basado en inteligencia artificial.

##  Licencia

Este proyecto fue desarrollado con fines de investigación y aplicación médica, en el marco del estudio DERMOSAN – UNDC 2025, orientado al apoyo diagnóstico dermatológico mediante inteligencia artificial.

##  Disclaimer Médico

Este sistema es una herramienta de apoyo al diagnóstico y NO reemplaza:
- El criterio clínico profesional
- La evaluación médica presencial  
- Los estudios complementarios necesarios
- La biopsia cuando esté indicada

Siempre consulte con un dermatólogo certificado para confirmación diagnóstica y tratamiento.

---

**Dermosan v1.0.0** - Sistema de Diagnóstico Dermatológico Automatizado  
*DERMOSAN – UNDC 2025*