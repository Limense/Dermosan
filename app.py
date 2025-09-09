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
            with st.spinner("🔄 Cargando modelo de IA..."):
                predictor = DermatologyPredictor()
                st.success("✅ Modelo cargado exitosamente")
                return predictor
        except FileNotFoundError as e:
            st.error("❌ **Error:** No se encontró el archivo del modelo")
            st.info("""
            📁 **Solución:** Asegúrese de que el archivo del modelo esté en una de estas ubicaciones:
            - `models/best_resnet152.h5` (recomendado)
            - `Modelo Entrenado/best_resnet152.h5` (alternativo)
            """)
            st.stop()
        except Exception as e:
            st.error(f"❌ **Error crítico al cargar el modelo:** {str(e)}")
            with st.expander("🔍 Detalles técnicos del error"):
                st.code(f"Tipo: {type(e).__name__}\nMensaje: {str(e)}")
            st.info("""
            🛠️ **Posibles soluciones:**
            1. Verificar que TensorFlow esté instalado: `pip install tensorflow`
            2. Comprobar compatibilidad del modelo
            3. Ejecutar script de verificación: `python verificar_modelo.py`
            """)
            return None
    
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    # Interfaz principal
    st.markdown("### 📸 Subir Imagen para Diagnóstico")
    
    # Métricas rápidas en la parte superior
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("🎯 Precisión", "94.2%", "↗ Optimizado")
    with col_b:
        st.metric("⚡ Velocidad", "< 3s", "↗ Rápido")
    with col_c:
        st.metric("🧠 Modelo", "ResNet152", "✅ Cargado")
    with col_d:
        st.metric("🏥 Estado", "Activo", "🟢 Online")
    
    st.markdown("---")
    
    # Instrucciones mejoradas con énfasis en imágenes médicas
    with st.expander("📋 Instrucciones Importantes de Uso", expanded=False):
        st.markdown("""
        ## 🏥 **IMPORTANTE: Solo Imágenes Dermatológicas Reales**
        
        ### ✅ **IMÁGENES PERMITIDAS:**
        - **📸 Fotografías directas de piel** tomadas con cámara
        - **🔍 Lesiones cutáneas visibles** (lunares, manchas, erupciones)
        - **📱 Fotos de alta calidad** enfocadas en la zona afectada
        - **💡 Buena iluminación natural** sin sombras excesivas
        
        ### ❌ **IMÁGENES PROHIBIDAS:**
        - **🚫 Capturas de pantalla** de cualquier tipo
        - **📄 Documentos, textos o PDFs** 
        - **🖥️ Interfaces de aplicaciones**
        - **🌐 Imágenes descargadas de internet**
        - **📊 Gráficos, diagramas o esquemas**
        
        ### 🎯 **Requisitos Técnicos:**
        1. **Resolución:** Mínimo 224x224 píxeles
        2. **Formato:** JPG, JPEG, PNG
        3. **Enfoque:** La lesión debe estar claramente visible
        4. **Iluminación:** Evite sombras o reflejos excesivos
        5. **Fondo:** Preferiblemente neutro
        
        ### ⚕️ **Consideraciones Médicas:**
        - Use este sistema solo como **herramienta de apoyo**
        - **Siempre consulte con un dermatólogo** para confirmación
        - Para casos urgentes, busque **atención médica inmediata**
        """)
    
    # Advertencia destacada sobre tipo de imágenes
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FF5722, #D32F2F); 
                color: white; 
                padding: 1.5rem; 
                border-radius: 10px; 
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 4px 8px rgba(255,87,34,0.3);">
        <h3 style="margin: 0; color: white;">
            🚨 ADVERTENCIA: SOLO FOTOGRAFÍAS MÉDICAS REALES
        </h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            El sistema rechazará automáticamente documentos, capturas de pantalla<br>
            y cualquier imagen que no sea una fotografía dermatológica real.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
                
                # Card para la imagen con diseño médico
                st.markdown("""
                <div style="background: white; 
                            border: 2px solid #E0E0E0; 
                            border-radius: 10px; 
                            padding: 1rem; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin-bottom: 1rem;">
                """, unsafe_allow_html=True)
                
                st.image(image, caption="Imagen para diagnóstico", use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Información de la imagen en cards
                st.markdown("#### 📋 Información Técnica")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                                color: white; 
                                padding: 1rem; 
                                border-radius: 8px; 
                                text-align: center;
                                margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; font-size: 1.1rem;">{image.format}</h4>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Formato</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4CAF50, #45a049); 
                                color: white; 
                                padding: 1rem; 
                                border-radius: 8px; 
                                text-align: center;
                                margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; font-size: 1.1rem;">{image.size[0]}×{image.size[1]}</h4>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Resolución</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Modo de color
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FF9800, #F57C00); 
                            color: white; 
                            padding: 1rem; 
                            border-radius: 8px; 
                            text-align: center;
                            margin-bottom: 1rem;">
                    <h4 style="margin: 0; font-size: 1.1rem;">{image.mode}</h4>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Modo de Color</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🔍 Análisis de Calidad")
                
                # Spinner personalizado para análisis
                with st.spinner("🧬 Analizando calidad de imagen con IA médica..."):
                    quality_result = analyze_image_quality(image)
                
                display_quality_analysis(quality_result)
                
                # Botón de análisis mejorado
                if not quality_result.get('is_suitable', False):
                    st.markdown("---")
                    st.markdown("""
                    <div style="background: #FFF3E0; 
                                border-left: 4px solid #FF9800; 
                                padding: 1rem; 
                                border-radius: 8px; 
                                margin: 1rem 0;">
                        <h4 style="color: #F57C00; margin: 0 0 0.5rem 0;">
                            ⚠️ Calidad de Imagen Subóptima
                        </h4>
                        <p style="margin: 0; color: #E65100;">
                            Se recomienda tomar una nueva fotografía siguiendo las instrucciones.
                            Sin embargo, puede proceder con el análisis bajo su propio criterio.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Realizar predicción solo si la imagen es válida y de calidad suficiente
            is_valid_medical = quality_result.get('is_medical_image', True)
            
            if not is_valid_medical:
                # Imagen no válida - mostrar solo error sin opción de análisis
                st.markdown("""
                <div style="background: #FFCDD2; 
                            border: 2px solid #F44336; 
                            padding: 2rem; 
                            border-radius: 15px; 
                            text-align: center;
                            margin: 2rem 0;
                            box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                    <h2 style="color: #C62828; margin: 0;">
                        🚫 ANÁLISIS BLOQUEADO POR SEGURIDAD
                    </h2>
                    <p style="color: #D32F2F; font-size: 1.2rem; margin: 1rem 0 0 0;">
                        El sistema ha detectado que esta imagen no es apropiada para análisis dermatológico.
                        Por favor, suba una fotografía real de piel o lesión cutánea.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            elif quality_result.get('is_suitable', False):
                # Imagen válida y de buena calidad - proceder automáticamente
                st.markdown("""
                <div style="background: #E8F5E8; 
                            border-left: 4px solid #4CAF50; 
                            padding: 1rem; 
                            border-radius: 8px; 
                            margin: 1rem 0;">
                    <h4 style="color: #2E7D32; margin: 0 0 0.5rem 0;">
                        ✅ Imagen Médica Válida y de Calidad Óptima
                    </h4>
                    <p style="margin: 0; color: #388E3C;">
                        La imagen ha pasado todas las validaciones. Procediendo con análisis dermatológico...
                    </p>
                </div>
                """, unsafe_allow_html=True)
                should_analyze = True
            else:
                # Imagen válida pero de calidad subóptima - permitir análisis con advertencia
                should_analyze = st.button(
                    "🔬 Proceder con Análisis (Calidad Subóptima)", 
                    type="secondary",
                    help="La imagen es médicamente válida pero de calidad subóptima. Proceder con precaución.",
                    use_container_width=True
                )
            
            if should_analyze:
                
                st.markdown("---")
                st.markdown("### 🎯 Resultados del Diagnóstico")
                
                with st.spinner("🧠 Analizando imagen con IA dermatológica avanzada..."):
                    prediction_result = predictor.predict(image)
                    recommendations = predictor.get_medical_recommendation(prediction_result)
                
                # Resultado principal con diseño médico mejorado
                predicted_disease = prediction_result['predicted_class']
                confidence = prediction_result['confidence']
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; 
                            padding: 2rem; 
                            border-radius: 15px; 
                            text-align: center;
                            margin: 1.5rem 0;
                            box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                    <h2 style="margin: 0; color: white; font-size: 1.8rem;">
                        DIAGNÓSTICO PRINCIPAL
                    </h2>
                    <h1 style="margin: 0.5rem 0; color: white; font-size: 2.2rem;">
                        {predicted_disease}
                    </h1>
                    <div style="background: rgba(255,255,255,0.2); 
                                padding: 0.8rem; 
                                border-radius: 20px; 
                                margin-top: 1rem; 
                                display: inline-block;">
                        <h3 style="margin: 0; font-size: 1.4rem;">
                            Confianza: {prediction_result['confidence_percentage']}
                        </h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Layout de resultados mejorado
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown("#### 📊 Análisis Detallado")
                    
                    # Gauge de confianza
                    st.plotly_chart(
                        create_confidence_gauge(confidence),
                        use_container_width=True
                    )
                    
                    # Información de la enfermedad
                    display_disease_info(predicted_disease)
                
                with result_col2:
                    st.markdown("#### 📈 Diagnósticos Diferenciales")
                    
                    # Top 3 predicciones con diseño mejorado
                    st.markdown("**🏆 Top 3 Diagnósticos Probables:**")
                    
                    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
                        # Color degradado según posición
                        colors = ["#4CAF50", "#FF9800", "#2196F3"]
                        color = colors[i-1]
                        
                        st.markdown(f"""
                        <div style="background: {color}15; 
                                    border-left: 4px solid {color}; 
                                    padding: 1rem; 
                                    margin: 0.5rem 0; 
                                    border-radius: 8px;">
                            <h4 style="color: {color}; margin: 0;">
                                #{i} {pred['disease']}
                            </h4>
                            <p style="margin: 0.3rem 0 0 0; font-size: 1.1rem; font-weight: bold;">
                                {pred['percentage']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Gráfico de probabilidades
                    st.plotly_chart(
                        create_probability_chart(prediction_result['all_probabilities']),
                        use_container_width=True
                    )
                
                # Recomendaciones médicas
                st.markdown("---")
                st.markdown("### 🏥 Recomendaciones Clínicas")
                display_medical_recommendations(recommendations)
                
                # Disclaimer médico con diseño prominente
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #FF5722, #D32F2F); 
                            color: white; 
                            padding: 2rem; 
                            border-radius: 15px; 
                            text-align: center;
                            margin: 1.5rem 0;
                            box-shadow: 0 4px 8px rgba(255,87,34,0.3);">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚕️</div>
                    <h2 style="margin: 0 0 1rem 0; color: white;">
                        ADVERTENCIA MÉDICA IMPORTANTE
                    </h2>
                    <p style="margin: 0; font-size: 1.2rem; line-height: 1.5;">
                        <strong>Este es un sistema de apoyo al diagnóstico.</strong><br>
                        Los resultados deben ser validados por un profesional médico calificado.<br>
                        <strong>NO reemplaza</strong> la evaluación clínica presencial ni el criterio médico profesional.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generar reporte con diseño mejorado
                st.markdown("---")
                st.markdown("### 📄 Generar Reporte Médico")
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                
                with col_btn2:
                    if st.button(
                        "� Generar Reporte Completo", 
                        type="primary",
                        use_container_width=True,
                        help="Genera un reporte médico detallado con todos los resultados"
                    ):
                        report = export_diagnosis_report(
                            image, prediction_result, quality_result, recommendations
                        )
                        
                        # Crear enlace de descarga
                        timestamp = prediction_result.get('timestamp', 'diagnosis')
                        filename = f"reporte_dermosan_{timestamp}.txt"
                        
                        st.success("✅ Reporte generado exitosamente")
                        
                        st.markdown(
                            create_download_link(report, filename, "📥 Descargar Reporte Médico"),
                            unsafe_allow_html=True
                        )
                        
                        # Mostrar preview del reporte
                        with st.expander("👁️ Vista previa del reporte"):
                            st.text(report)
            
            elif not quality_result.get('is_suitable', False):
                st.markdown("""
                <div style="background: #FFEBEE; 
                            border-left: 4px solid #F44336; 
                            padding: 1.5rem; 
                            border-radius: 8px; 
                            margin: 1rem 0;">
                    <h4 style="color: #D32F2F; margin: 0 0 0.5rem 0;">
                        ⚠️ Calidad de Imagen No Óptima
                    </h4>
                    <p style="margin: 0; color: #C62828;">
                        La calidad de la imagen no es óptima para un diagnóstico confiable.<br>
                        <strong>Se recomienda tomar una nueva fotografía siguiendo las instrucciones.</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
            logging.error(f"Error en procesamiento: {str(e)}")
            
            # Mostrar información de ayuda
            with st.expander("🔍 Información del error"):
                st.markdown(f"""
                **Tipo de error:** `{type(e).__name__}`  
                **Mensaje:** {str(e)}  
                **Archivo:** {uploaded_file.name if uploaded_file else 'No especificado'}  
                """)
                
            st.info("""
            💡 **Sugerencias:**
            - Verifique que el archivo sea una imagen válida (JPG, PNG)
            - Asegúrese de que la imagen no esté corrupta
            - Intente con una imagen diferente
            - La imagen debe tener al menos 224x224 píxeles
            """)
    
    else:
        # Mostrar información cuando no hay imagen cargada
        st.info("👆 Suba una imagen dermatológica para comenzar el análisis")
        
        # Mostrar ejemplos o información adicional
        st.markdown("---")
        
        # Añadir pestaña de información
        tab1, tab2, tab3 = st.tabs(["📚 Enfermedades", "📊 Métricas del Modelo", "ℹ️ Información"])
        
        with tab1:
            st.markdown("### 🎯 Enfermedades que puede diagnosticar el sistema:")
            
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
        
        with tab2:
            from src.utils import display_model_metrics
            display_model_metrics()
        
        with tab3:
            st.markdown("### ℹ️ Acerca de Dermosan")
            st.markdown("""
            **Dermosan** es un sistema de inteligencia artificial desarrollado específicamente 
            para las Clínicas de San Vicente en Cañete, diseñado para asistir a los profesionales 
            médicos en el diagnóstico de enfermedades dermatológicas.
            
            #### 🔬 Tecnología
            - **Modelo:** ResNet152 con Transfer Learning
            - **Framework:** TensorFlow 2.15+
            - **Interfaz:** Streamlit
            - **Precisión:** 94.2% en conjunto de prueba
            
            #### ⚕️ Uso Médico
            - Herramienta de **apoyo al diagnóstico**
            - **NO reemplaza** el criterio médico profesional
            - Requiere **validación** por dermatólogo certificado
            - Ideal para screening y segunda opinión
            
            #### 📞 Soporte
            Para soporte técnico o consultas médicas, contacte con:
            **Clínicas de San Vicente, Cañete**
            """)
            
            # Agregar información de versión y fecha
            from datetime import datetime
            st.markdown(f"""
            ---
            **Versión:** 1.0.0  
            **Última actualización:** {datetime.now().strftime("%d/%m/%Y")}  
            **Desarrollado por:** Equipo de IA Médica
            """)

if __name__ == "__main__":
    main()
