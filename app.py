"""
Aplicación principal del sistema de diagnóstico dermatológico Dermosan
"""

import streamlit as st
import logging
from PIL import Image
import io
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Agregar el directorio actual al path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports de módulos locales
from src.predictor import DermatologyPredictor, analyze_image_quality
from src.utils import (
    set_page_config, display_header, display_sidebar_info,
    create_confidence_gauge, create_probability_chart,
    display_disease_info, display_quality_analysis,
    display_medical_recommendations, export_diagnosis_report,
    create_download_link
)
from src.config import APP_CONFIG

def main():
    """Función principal de la aplicación."""
    
    # Configurar página
    set_page_config()
    
    # Mostrar header
    display_header()
    
    # Mostrar información en sidebar
    display_sidebar_info()
    
    # Inicializar predictor
    @st.cache_resource
    def load_predictor():
        """Carga el predictor con cache para optimizar rendimiento."""
        try:
            return DermatologyPredictor()
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            st.info("Asegúrese de que el archivo 'best_resnet152.h5' esté en la carpeta 'Modelo Entrenado'")
            return None
    
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    # Interfaz principal
    st.markdown("### 📸 Subir Imagen para Diagnóstico")
    
    # Instrucciones
    with st.expander("📋 Instrucciones de uso", expanded=False):
        st.markdown("""
        **Para obtener mejores resultados:**
        
        1. **Calidad de imagen:** Use imágenes nítidas y bien iluminadas
        2. **Resolución:** Mínimo 224x224 píxeles
        3. **Enfoque:** La lesión debe estar claramente visible
        4. **Iluminación:** Evite sombras o reflejos excesivos
        5. **Fondo:** Preferiblemente fondo neutro
        
        **Formatos aceptados:** JPG, JPEG, PNG
        """)
    
    # Upload de imagen
    uploaded_file = st.file_uploader(
        "Seleccione una imagen dermatológica",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            # Cargar imagen
            image = Image.open(uploaded_file)
            
            # Layout con columnas
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🖼️ Imagen Cargada")
                st.image(image, caption="Imagen para diagnóstico", use_column_width=True)
                
                # Información de la imagen
                st.markdown(f"""
                **Información de la imagen:**
                - Formato: {image.format}
                - Tamaño: {image.size[0]} x {image.size[1]} px
                - Modo: {image.mode}
                """)
            
            with col2:
                st.markdown("#### 🔍 Análisis de Calidad")
                
                # Analizar calidad de imagen
                with st.spinner("Analizando calidad de imagen..."):
                    quality_result = analyze_image_quality(image)
                
                display_quality_analysis(quality_result)
            
            # Realizar predicción si la imagen es de calidad suficiente
            if quality_result.get('is_suitable', False) or st.button("🔬 Analizar de todas formas"):
                
                st.markdown("---")
                st.markdown("### 🎯 Resultados del Diagnóstico")
                
                with st.spinner("Analizando imagen con inteligencia artificial..."):
                    prediction_result = predictor.predict(image)
                    recommendations = predictor.get_medical_recommendation(prediction_result)
                
                # Layout de resultados
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown("#### 📊 Diagnóstico Principal")
                    
                    # Mostrar resultado principal
                    predicted_disease = prediction_result['predicted_class']
                    confidence = prediction_result['confidence']
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                        <h3 style="margin: 0; color: white;">{predicted_disease}</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2em;">
                            Confianza: {prediction_result['confidence_percentage']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge de confianza
                    st.plotly_chart(
                        create_confidence_gauge(confidence),
                        use_container_width=True
                    )
                    
                    # Información de la enfermedad
                    display_disease_info(predicted_disease)
                
                with result_col2:
                    st.markdown("#### 📈 Análisis Completo")
                    
                    # Top 3 predicciones
                    st.markdown("**Top 3 Diagnósticos Diferenciales:**")
                    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
                        st.markdown(f"{i}. **{pred['disease']}** - {pred['percentage']}")
                    
                    # Gráfico de probabilidades
                    st.plotly_chart(
                        create_probability_chart(prediction_result['all_probabilities']),
                        use_container_width=True
                    )
                
                # Recomendaciones médicas
                st.markdown("---")
                st.markdown("### 🏥 Recomendaciones Clínicas")
                display_medical_recommendations(recommendations)
                
                # Disclaimer médico
                st.markdown("---")
                st.warning("""
                ⚠️ **IMPORTANTE:** Este es un sistema de apoyo al diagnóstico. 
                Los resultados deben ser validados por un profesional médico calificado. 
                No reemplaza la evaluación clínica presencial ni el criterio médico profesional.
                """)
                
                # Generar reporte
                st.markdown("---")
                st.markdown("### 📄 Generar Reporte")
                
                if st.button("🔗 Generar Reporte Completo"):
                    report = export_diagnosis_report(
                        image, prediction_result, quality_result, recommendations
                    )
                    
                    # Crear enlace de descarga
                    timestamp = prediction_result.get('timestamp', 'diagnosis')
                    filename = f"reporte_dermosan_{timestamp}.txt"
                    
                    st.markdown(
                        create_download_link(report, filename, "📥 Descargar Reporte"),
                        unsafe_allow_html=True
                    )
                    
                    # Mostrar preview del reporte
                    with st.expander("👁️ Vista previa del reporte"):
                        st.text(report)
            
            elif not quality_result.get('is_suitable', False):
                st.warning("""
                ⚠️ La calidad de la imagen no es óptima para un diagnóstico confiable. 
                Se recomienda tomar una nueva fotografía siguiendo las instrucciones.
                """)
        
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")
            logging.error(f"Error en procesamiento: {str(e)}")
    
    else:
        # Mostrar información cuando no hay imagen cargada
        st.info("👆 Suba una imagen dermatológica para comenzar el análisis")
        
        # Mostrar ejemplos o información adicional
        st.markdown("---")
        st.markdown("### 📚 Enfermedades que puede diagnosticar el sistema:")
        
        diseases_info = [
            "🔴 **Melanoma** - Cáncer de piel maligno",
            "🟠 **Carcinoma Basocelular** - Cáncer de piel común",
            "🟡 **Eczema** - Inflamación crónica de la piel",
            "🟢 **Nevos Melanocíticos** - Lunares benignos",
            "🔵 **Dermatitis Atópica** - Eczema atópico",
            "🟣 **Psoriasis** - Enfermedad inflamatoria crónica",
            "🟤 **Queratosis Seborreica** - Lesiones benignas",
            "⚪ **Queratosis Benigna** - Lesiones queratósicas",
            "🔶 **Infecciones Virales** - Verrugas y molluscum",
            "🔸 **Infecciones Fúngicas** - Tiña y candidiasis"
        ]
        
        cols = st.columns(2)
        for i, disease in enumerate(diseases_info):
            with cols[i % 2]:
                st.markdown(disease)

if __name__ == "__main__":
    main()
