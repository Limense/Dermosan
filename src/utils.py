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
    """Muestra el header de la aplicación."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🏥 Dermosan - Sistema de Diagnóstico Dermatológico
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            Clínicas de San Vicente, Cañete | Versión 1.0.0
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_info():
    """Muestra información en la barra lateral."""
    with st.sidebar:
        st.markdown("### ℹ️ Información del Sistema")
        st.markdown("""
        **Modelo:** ResNet152  
        **Precisión:** ~94.2%  
        **Clases:** 10 enfermedades  
        **Dataset:** 30,000+ imágenes  
        """)
        
        st.markdown("### 📊 Estadísticas del Dataset")
        dataset_stats = {
            "Melanocytic Nevi": 7970,
            "Basal Cell Carcinoma": 3323,
            "Benign Keratosis": 2624,
            "Warts Molluscum": 2103,
            "Psoriasis Lichen": 2000,
            "Seborrheic Keratoses": 1800,
            "Tinea Ringworm": 1700,
            "Eczema": 1677,
            "Atopic Dermatitis": 1250,
            "Melanoma": 15750
        }
        
        df_stats = pd.DataFrame(
            list(dataset_stats.items()), 
            columns=['Enfermedad', 'Imágenes']
        )
        st.dataframe(df_stats, hide_index=True)

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
    Muestra el análisis de calidad de imagen.
    
    Args:
        quality_result: Resultado del análisis de calidad
    """
    score = quality_result["quality_score"]
    
    # Determinar color y mensaje
    if score >= 75:
        color = "green"
        message = "✅ Imagen de buena calidad para diagnóstico"
    elif score >= 50:
        color = "orange" 
        message = "⚠️ Calidad aceptable, pero podría mejorarse"
    else:
        color = "red"
        message = "❌ Calidad insuficiente para diagnóstico confiable"
    
    st.markdown(f"""
    <div style="background-color: {color}15; 
                border: 1px solid {color}; 
                padding: 1rem; margin: 1rem 0; border-radius: 5px;">
        <h4 style="color: {color}; margin: 0;">{message}</h4>
        <p><strong>Puntuación de calidad:</strong> {score}/100</p>
        <p><strong>Resolución:</strong> {quality_result.get('resolution', 'N/A')}</p>
    """, unsafe_allow_html=True)
    
    if quality_result.get("issues"):
        st.markdown("<strong>Problemas detectados:</strong>")
        for issue in quality_result["issues"]:
            st.markdown(f"• {issue}")
    
    st.markdown("</div>", unsafe_allow_html=True)

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
