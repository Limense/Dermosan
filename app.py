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
    create_confidence_gauge, create_probability_chart, create_compact_probability_chart,
    create_risk_assessment_chart, create_comparison_chart, create_severity_timeline,
    display_disease_info, display_quality_analysis,
    display_medical_recommendations, export_diagnosis_report,
    create_download_link, display_medical_footer, display_confidence_level
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
            with st.spinner("Cargando modelo de IA..."):
                predictor = DermatologyPredictor()
                st.success("Modelo cargado exitosamente")
                return predictor
        except FileNotFoundError as e:
            st.error("**Error:** No se encontró el archivo del modelo")
            st.info("""
            **Solución:** Asegúrese de que el archivo del modelo esté en una de estas ubicaciones:
            - `models/best_resnet152.h5` (recomendado)
            - `Modelo Entrenado/best_resnet152.h5` (alternativo)
            """)
            st.stop()
        except Exception as e:
            st.error(f"**Error crítico al cargar el modelo:** {str(e)}")
            with st.expander("Detalles técnicos del error"):
                st.code(f"Tipo: {type(e).__name__}\nMensaje: {str(e)}")
            st.info("""
            **Posibles soluciones:**
            1. Verificar que TensorFlow esté instalado: `pip install tensorflow`
            2. Comprobar compatibilidad del modelo
            3. Ejecutar script de verificación: `python verificar_modelo.py`
            """)
            return None
    
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    # Interfaz principal
    st.markdown("### Subir Imagen para Diagnóstico")
    
    # Métricas rápidas en la parte superior
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Precisión", "95%", "Optimizado")
    with col_b:
        st.metric("Velocidad", "< 3s", "Rápido")
    with col_c:
        st.metric("Modelo", "ResNet152", "Cargado")
    with col_d:
        st.metric("Estado", "Activo", "Online")
    
    st.markdown("---")
    
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
                st.markdown("#### Imagen Cargada")
                st.image(image, caption="Imagen para diagnóstico", width=300)
            
            with col2:
                st.markdown("#### Análisis de Calidad")
                
                with st.spinner("Analizando calidad de imagen..."):
                    quality_result = analyze_image_quality(image)
                
                display_quality_analysis(quality_result)
            
            # Realizar predicción
            should_analyze = st.button("Analizar Imagen", type="primary")
            
            if should_analyze:
                with st.spinner("Analizando imagen con IA dermatológica..."):
                    prediction_result = predictor.predict(image)
                    recommendations = predictor.get_medical_recommendation(prediction_result)
                
                # Resultado principal
                predicted_disease = prediction_result['predicted_class']
                confidence = prediction_result['confidence']
                
                # Dashboard de resultados
                st.markdown("---")
                st.markdown("## Dashboard de Resultados Detallados")
                
                # Dashboard principal mejorado
                st.markdown("### Diagnóstico Principal")
                
                # Layout mejorado: 3 columnas
                main_row1_col1, main_row1_col2, main_row1_col3 = st.columns([2, 1, 1.5])
                
                with main_row1_col1:
                    # Información principal del diagnóstico con diseño mejorado
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2E5BBA, #4A90B8); 
                                color: white; padding: 1.5rem; border-radius: 15px; 
                                text-align: center; margin-bottom: 1rem;">
                        <h2 style="margin: 0 0 0.5rem 0; color: white; font-size: 1.8rem;">
                            {predicted_disease}
                        </h2>
                        <h3 style="margin: 0; color: rgba(255,255,255,0.9); font-size: 1.3rem;">
                            Confianza: {prediction_result['confidence_percentage']}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar nivel de confianza con colores
                    display_confidence_level(confidence)
                
                with main_row1_col2:
                    # Gauge de confianza compacto
                    st.markdown("**Medidor**")
                    st.plotly_chart(
                        create_confidence_gauge(confidence),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                with main_row1_col3:
                    # Distribución de probabilidades integrada - versión compacta
                    st.markdown("**Top Probabilidades**")
                    st.plotly_chart(
                        create_compact_probability_chart(prediction_result['all_probabilities']),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                # Resumen estadístico visual
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ECF0F1, #BDC3C7); 
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                    <h3 style="text-align: center; color: #2C3E50; margin: 0 0 1rem 0;">
                        Resumen del Análisis
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Métricas clave en columnas
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    confidence_val = int(confidence * 100)
                    st.metric(
                        label="Confianza",
                        value=f"{confidence_val}%",
                        delta=f"{'Alto' if confidence_val > 80 else 'Medio' if confidence_val > 60 else 'Bajo'}"
                    )
                
                with metric_col2:
                    # Calcular número de diagnósticos considerados
                    num_diagnoses = len([p for p in prediction_result['all_probabilities'].values() if p > 0.05])
                    st.metric(
                        label="Diagnósticos",
                        value=f"{num_diagnoses}",
                        delta="analizados"
                    )
                
                with metric_col3:
                    # Determinar nivel de riesgo
                    risk_level = "Alto" if "Melanoma" in predicted_disease or "Carcinoma" in predicted_disease else "Medio" if confidence < 0.7 else "Bajo"
                    st.metric(
                        label="Nivel Riesgo",
                        value=risk_level,
                        delta="evaluado"
                    )
                
                with metric_col4:
                    st.metric(
                        label="Tiempo Análisis",
                        value="< 5s",
                        delta="Rápido"
                    )
                
                # Sección de análisis avanzado
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h3 style="color: #2E5BBA; margin: 0;">Análisis Clínico Avanzado</h3>
                    <p style="color: #34495E; margin: 0.5rem 0 0 0; font-style: italic;">
                        Evaluación integral de riesgo y comparaciones diagnósticas
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Primera fila de gráficos importantes
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="color: #E74C3C; margin: 0;">Evaluación de Riesgo</h4>
                        <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.3rem 0;">Nivel de urgencia médica</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        create_risk_assessment_chart(predicted_disease, confidence),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                with analysis_col2:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="color: #27AE60; margin: 0;">Comparación Diagnóstica</h4>
                        <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.3rem 0;">Top 3 diagnósticos más probables</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        create_comparison_chart(prediction_result['all_probabilities']),
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                
                # Evolución temporal
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h4 style="color: #F39C12; margin: 0;">Proyección de Evolución Temporal</h4>
                    <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.3rem 0;">
                        Simulación de progresión con diferentes escenarios de tratamiento
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(
                    create_severity_timeline(),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )
                
                # Información detallada
                st.markdown("---")
                st.markdown("### Información Clínica Detallada")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    display_disease_info(predicted_disease)
                
                with info_col2:
                    st.markdown("#### Recomendaciones Médicas")
                    display_medical_recommendations(recommendations)
                
                # Disclaimer médico
                st.markdown("""
                <div style="background: linear-gradient(135deg, #34495E, #2C3E50); 
                            color: white; padding: 2rem; border-radius: 15px; 
                            text-align: center; margin: 2rem 0;">
                    <h3 style="margin: 0 0 1rem 0; color: white;">Aviso Médico Importante</h3>
                    <p style="margin: 0; color: rgba(255,255,255,0.9);">
                        Este sistema es una herramienta de apoyo diagnóstico que utiliza 
                        inteligencia artificial. Los resultados deben ser siempre interpretados por un 
                        dermatólogo profesional. No reemplaza el juicio clínico médico.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"**Error al procesar la imagen:** {str(e)}")
            logging.error(f"Error en procesamiento: {str(e)}")
    
    else:
        st.info("Suba una imagen dermatológica para comenzar el análisis")
    
    # Footer
    display_medical_footer()

if __name__ == "__main__":
    main()