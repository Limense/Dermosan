"""
Utilidades para el sistema de diagnóstico dermatológico
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import base64
import io

from src.config import DISEASE_INFO, APP_CONFIG

def set_page_config():
    """Configura la página de Streamlit."""
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def display_header():
    """Muestra el header de la aplicación con información mejorada."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            🏥 Dermosan - Sistema de Diagnóstico Dermatológico
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Clínicas de San Vicente, Cañete | Versión 1.0.0
        </p>
        <div style="text-align: center; margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.3rem 1rem; 
                         border-radius: 20px; font-size: 0.9rem;">
                🤖 IA ResNet152 | 📊 94.2% Precisión | 🎯 10 Enfermedades
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_info():
    """Muestra información mejorada en la barra lateral."""
    with st.sidebar:
        st.markdown("### ℹ️ Información del Sistema")
        
        # Métricas del modelo en cajas coloridas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                        padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 0.5rem;">
                <h3 style="margin: 0; font-size: 1.5rem;">94.2%</h3>
                <p style="margin: 0; font-size: 0.8rem;">Precisión</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF6B6B, #FF8E53); color: white; 
                        padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 0.5rem;">
                <h3 style="margin: 0; font-size: 1.5rem;">10</h3>
                <p style="margin: 0; font-size: 0.8rem;">Enfermedades</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **Modelo:** ResNet152 con Transfer Learning  
        **Dataset:** 30,000+ imágenes médicas  
        **Input:** 224x224 RGB  
        **Tiempo promedio:** < 3 segundos  
        """)
        
        # Agregar estado del sistema
        st.markdown("### 🔋 Estado del Sistema")
        st.success("🟢 Sistema Operativo")
        st.info("🔄 Modelo Cargado")
        st.info("📡 IA Lista para Diagnóstico")
        
        st.markdown("### 📊 Estadísticas del Dataset")
        dataset_stats = {
            "Melanoma": 15750,
            "Melanocytic Nevi": 7970,
            "Basal Cell Carcinoma": 3323,
            "Benign Keratosis": 2624,
            "Warts Molluscum": 2103,
            "Psoriasis Lichen": 2000,
            "Seborrheic Keratoses": 1800,
            "Tinea Ringworm": 1700,
            "Eczema": 1677,
            "Atopic Dermatitis": 1250
        }
        
        df_stats = pd.DataFrame(
            list(dataset_stats.items()), 
            columns=['Enfermedad', 'Imágenes']
        )
        st.dataframe(df_stats, hide_index=True, use_container_width=True)

def create_confidence_gauge(confidence: float) -> go.Figure:
    """
    Crea un gauge chart para mostrar la confianza.
    
    Args:
        confidence: Valor de confianza (0-1)
        
    Returns:
        Figura de Plotly
    """
    confidence_percentage = confidence * 100
    
    # Determinar color basado en confianza
    if confidence >= 0.8:
        color = "green"
    elif confidence >= 0.6:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_percentage,
        title={'text': "Nivel de Confianza (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Crea un gráfico de barras con las probabilidades de cada clase.
    
    Args:
        probabilities: Diccionario con probabilidades por clase
        
    Returns:
        Figura de Plotly
    """
    # Preparar datos
    diseases = list(probabilities.keys())
    probs = [probabilities[disease] * 100 for disease in diseases]
    
    # Verificar que tenemos datos
    if not probs:
        # Crear gráfico vacío si no hay datos
        fig = go.Figure()
        fig.update_layout(title="No hay datos para mostrar")
        return fig
    
    # Obtener colores de configuración
    colors = [DISEASE_INFO.get(disease, {}).get("color", "#636EFA") for disease in diseases]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=diseases,
            orientation='h',
            marker_color=colors,
            text=[f"{prob:.1f}%" for prob in probs],
            textposition='outside'
        )
    ])
    
    # Calcular rango del eje X de manera segura
    max_prob = max(probs) if probs else 100
    x_range = [0, max_prob * 1.1]
    
    fig.update_layout(
        title="Probabilidades por Enfermedad",
        xaxis_title="Probabilidad (%)",
        yaxis_title="Enfermedad",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=x_range)
    )
    
    return fig

def display_disease_info(disease_name: str):
    """
    Muestra información detallada de una enfermedad.
    
    Args:
        disease_name: Nombre de la enfermedad
    """
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO[disease_name]
        
        # Determinar emoji según severidad
        severity_emoji = {
            "Benigna": "✅",
            "Leve": "💛", 
            "Leve a Moderada": "🟡",
            "Moderada": "🟠",
            "Grave - Requiere atención inmediata": "🚨"
        }
        
        emoji = severity_emoji.get(info["severity"], "ℹ️")
        
        st.markdown(f"""
        <div style="background-color: {info['color']}15; 
                    border-left: 4px solid {info['color']}; 
                    padding: 1rem; margin: 1rem 0; border-radius: 5px;">
            <h4 style="margin: 0; color: {info['color']};">
                {emoji} {disease_name}
            </h4>
            <p><strong>Descripción:</strong> {info['description']}</p>
            <p><strong>Severidad:</strong> {info['severity']}</p>
            <p><strong>Tratamiento típico:</strong> {info['treatment']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_quality_analysis(quality_result: Dict):
    """
    Muestra el análisis de calidad de imagen con validación médica mejorada.
    
    Args:
        quality_result: Resultado del análisis de calidad
    """
    # 1. VALIDACIÓN CRÍTICA: Verificar si es imagen médica válida
    if not quality_result.get("is_medical_image", True):
        error_type = quality_result.get("validation_error", "unknown")
        
        # Mostrar error crítico según el tipo
        if error_type == "text_document":
            st.markdown("""
            <div style="background: #FFEBEE; 
                        border: 3px solid #F44336; 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                <div style="font-size: 4rem; color: #F44336; margin-bottom: 1rem;">🚫</div>
                <h2 style="color: #C62828; margin: 0 0 1rem 0;">
                    DOCUMENTO DETECTADO
                </h2>
                <p style="color: #D32F2F; font-size: 1.3rem; margin: 0; line-height: 1.5;">
                    <strong>Esta imagen contiene texto/documento, no una fotografía médica.</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif error_type == "screenshot":
            st.markdown("""
            <div style="background: #FFEBEE; 
                        border: 3px solid #F44336; 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                <div style="font-size: 4rem; color: #F44336; margin-bottom: 1rem;">📱</div>
                <h2 style="color: #C62828; margin: 0 0 1rem 0;">
                    CAPTURA DE PANTALLA DETECTADA
                </h2>
                <p style="color: #D32F2F; font-size: 1.3rem; margin: 0; line-height: 1.5;">
                    <strong>No se permiten capturas de pantalla.</strong><br>
                    Use una cámara para fotografiar directamente la piel.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        elif error_type == "not_medical":
            st.markdown("""
            <div style="background: #FFEBEE; 
                        border: 3px solid #F44336; 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(244,67,54,0.3);">
                <div style="font-size: 4rem; color: #F44336; margin-bottom: 1rem;">🏥</div>
                <h2 style="color: #C62828; margin: 0 0 1rem 0;">
                    IMAGEN NO MÉDICA
                </h2>
                <p style="color: #D32F2F; font-size: 1.3rem; margin: 0; line-height: 1.5;">
                    <strong>Esta imagen no parece ser de piel o lesión cutánea.</strong><br>
                    Sistema diseñado solo para imágenes dermatológicas.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar problemas específicos
        if quality_result.get("issues"):
            st.markdown("### ❌ **Problemas Detectados:**")
            for issue in quality_result["issues"]:
                st.markdown(f"""
                <div style="background: #FFCDD2; 
                            border-left: 4px solid #F44336; 
                            padding: 1rem; 
                            margin: 0.5rem 0; 
                            border-radius: 4px;">
                    <strong style="color: #B71C1C;">{issue}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        # Instrucciones para imagen válida
        st.markdown("""
        <div style="background: #E8F5E8; 
                    border-left: 4px solid #4CAF50; 
                    padding: 2rem; 
                    margin: 1.5rem 0; 
                    border-radius: 8px;">
            <h3 style="color: #2E7D32; margin: 0 0 1rem 0;">
                📸 Cómo tomar una fotografía dermatológica correcta:
            </h3>
            <div style="color: #388E3C;">
                <p><strong>✅ Use fotografía directa con cámara</strong></p>
                <p><strong>✅ Enfoque la piel o lesión claramente</strong></p>
                <p><strong>✅ Buena iluminación natural</strong></p>
                <p><strong>✅ Distancia apropiada (no muy cerca/lejos)</strong></p>
                <p><strong>✅ Fondo neutro sin distracciones</strong></p>
                <br>
                <p><strong>❌ NO use capturas de pantalla</strong></p>
                <p><strong>❌ NO suba documentos o textos</strong></p>
                <p><strong>❌ NO use imágenes de internet</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return  # Salir sin mostrar análisis de calidad
    
    # 2. Si es imagen médica válida, mostrar análisis normal
    score = quality_result["quality_score"]
    
    # Determinar color y mensaje
    if score >= 75:
        color = "#4CAF50"
        bg_color = "#E8F5E8"
        icon = "✅"
        status = "ÓPTIMA"
        message = "Imagen de buena calidad para diagnóstico"
    elif score >= 50:
        color = "#FF9800"
        bg_color = "#FFF3E0"
        icon = "⚠️"
        status = "ACEPTABLE"
        message = "Calidad aceptable, pero podría mejorarse"
    else:
        color = "#F44336"
        bg_color = "#FFEBEE"
        icon = "❌"
        status = "INSUFICIENTE"
        message = "Calidad insuficiente para diagnóstico confiable"
    
    # Card principal de calidad
    st.markdown(f"""
    <div style="background: {bg_color}; 
                border-left: 4px solid {color}; 
                padding: 1.5rem; 
                margin: 1rem 0; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</div>
            <h3 style="color: {color}; margin: 0; font-size: 1.2rem;">
                CALIDAD DE IMAGEN: {status}
            </h3>
        </div>
        <p style="margin: 0.5rem 0; font-size: 1rem; color: #333;">
            <strong>{message}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas técnicas en cards pequeñas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; 
                    border: 1px solid #E0E0E0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: {color}; margin: 0; font-size: 1.5rem;">{score}</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #666;">
                Puntuación / 100
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        resolution = quality_result.get('resolution', 'N/A')
        st.markdown(f"""
        <div style="background: white; 
                    border: 1px solid #E0E0E0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: #2196F3; margin: 0; font-size: 1.2rem;">{resolution}</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #666;">
                Resolución
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Indicador de idoneidad
        suitable = quality_result.get('is_suitable', False)
        suitable_text = "APTA" if suitable else "NO APTA"
        suitable_color = "#4CAF50" if suitable else "#F44336"
        
        st.markdown(f"""
        <div style="background: white; 
                    border: 1px solid #E0E0E0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h4 style="color: {suitable_color}; margin: 0; font-size: 1.2rem;">{suitable_text}</h4>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #666;">
                Para diagnóstico
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar problemas detectados si los hay
    if quality_result.get("issues"):
        st.markdown("---")
        st.markdown("### 🔍 **Análisis Técnico Detallado**")
        
        issues_html = ""
        for issue in quality_result["issues"]:
            issues_html += f"""
            <div style="background: #FFF3E0; 
                        border-left: 3px solid #FF9800; 
                        padding: 0.8rem; 
                        margin: 0.5rem 0; 
                        border-radius: 4px;">
                <span style="color: #F57C00;">⚠️</span> 
                <strong style="color: #E65100;">{issue}</strong>
            </div>
            """
        
        st.markdown(issues_html, unsafe_allow_html=True)
        
        # Recomendaciones para mejorar
        st.markdown("""
        <div style="background: #E3F2FD; 
                    border-left: 3px solid #2196F3; 
                    padding: 1rem; 
                    margin: 1rem 0; 
                    border-radius: 4px;">
            <h4 style="color: #1976D2; margin: 0 0 0.5rem 0;">
                💡 Recomendaciones para Mejorar la Imagen:
            </h4>
            <ul style="margin: 0; color: #1565C0;">
                <li>Use iluminación natural o luz blanca uniforme</li>
                <li>Mantenga la cámara estable para evitar desenfoque</li>
                <li>Asegúrese de que la lesión esté bien enfocada</li>
                <li>Use una resolución mínima de 224x224 píxeles</li>
                <li>Evite sombras y reflejos en la imagen</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_medical_recommendations(recommendations: Dict):
    """
    Muestra las recomendaciones médicas.
    
    Args:
        recommendations: Diccionario con recomendaciones
    """
    urgency = recommendations["urgency"]
    
    # Determinar color según urgencia
    if "URGENTE" in urgency.upper():
        color = "#FF4757"
        icon = "🚨"
    elif "Prioritario" in urgency:
        color = "#FFA726"
        icon = "⚠️"
    else:
        color = "#42A5F5"
        icon = "ℹ️"
    
    st.markdown(f"""
    <div style="background-color: {color}15; 
                border-left: 4px solid {color}; 
                padding: 1.5rem; margin: 1rem 0; border-radius: 5px;">
        <h4 style="color: {color}; margin: 0;">
            {icon} Recomendaciones Médicas
        </h4>
        <p><strong>Urgencia:</strong> {urgency}</p>
        <p><strong>Acción recomendada:</strong> {recommendations['recommended_action']}</p>
        <p><strong>Seguimiento:</strong> {recommendations['follow_up']}</p>
    </div>
    """, unsafe_allow_html=True)

def export_diagnosis_report(image, prediction_result: Dict, quality_result: Dict, recommendations: Dict) -> str:
    """
    Genera un reporte de diagnóstico en formato texto.
    
    Args:
        image: Imagen analizada
        prediction_result: Resultado de predicción
        quality_result: Resultado de análisis de calidad
        recommendations: Recomendaciones médicas
        
    Returns:
        String con el reporte
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
    REPORTE DE DIAGNÓSTICO DERMATOLÓGICO AUTOMATIZADO
    =================================================
    
    Fecha y hora: {timestamp}
    Sistema: Dermosan v1.0.0
    Clínica: San Vicente, Cañete
    
    ANÁLISIS DE IMAGEN:
    ------------------
    Calidad de imagen: {quality_result['quality_score']}/100
    Resolución: {quality_result.get('resolution', 'N/A')}
    Adecuada para diagnóstico: {'Sí' if quality_result['is_suitable'] else 'No'}
    
    RESULTADOS DE DIAGNÓSTICO:
    -------------------------
    Diagnóstico principal: {prediction_result['predicted_class']}
    Confianza: {prediction_result['confidence_percentage']} ({prediction_result['confidence_level']})
    
    TOP 3 DIAGNÓSTICOS DIFERENCIALES:
    """
    
    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
        report += f"\n    {i}. {pred['disease']} - {pred['percentage']}"
    
    report += f"""
    
    RECOMENDACIONES CLÍNICAS:
    ------------------------
    Urgencia: {recommendations['urgency']}
    Acción recomendada: {recommendations['recommended_action']}
    Seguimiento: {recommendations['follow_up']}
    
    IMPORTANTE:
    ----------
    Este diagnóstico es generado por inteligencia artificial y debe ser
    validado por un profesional médico calificado. No reemplaza el
    criterio clínico ni la evaluación presencial.
    
    """
    
    return report

def display_model_metrics():
    """Muestra métricas detalladas del modelo."""
    st.markdown("### 📈 Métricas del Modelo ResNet152")
    
    # Métricas en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50, #45a049); color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">94.2%</h2>
            <p style="margin: 0.5rem 0 0 0;">Precisión General</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2196F3, #1976D2); color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">30K+</h2>
            <p style="margin: 0.5rem 0 0 0;">Imágenes Entrenamiento</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9800, #F57C00); color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">ResNet152</h2>
            <p style="margin: 0.5rem 0 0 0;">Arquitectura Avanzada</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Información técnica
    st.markdown("#### 🔧 Detalles Técnicos")
    tech_info = {
        "Arquitectura": "ResNet152 + Transfer Learning",
        "Capas Entrenables": "Últimas 50 capas",
        "Optimizador": "Adam con learning rate adaptativo",
        "Función de Pérdida": "Sparse Categorical Crossentropy",
        "Augmentación": "Rotación, zoom, flip horizontal",
        "Balanceamiento": "Pesos de clase automático",
        "Validación": "Split 80/10/10 (train/val/test)",
        "Tiempo de Entrenamiento": "~6 horas en GPU"
    }
    
    tech_df = pd.DataFrame(list(tech_info.items()), columns=['Parámetro', 'Valor'])
    st.dataframe(tech_df, hide_index=True, use_container_width=True)

def create_download_link(content: str, filename: str, link_text: str) -> str:
    """
    Crea un link de descarga para contenido de texto.
    
    Args:
        content: Contenido a descargar
        filename: Nombre del archivo
        link_text: Texto del enlace
        
    Returns:
        HTML del enlace de descarga
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href
