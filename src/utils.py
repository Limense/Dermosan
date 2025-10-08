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
from src.styles import apply_custom_styling, create_metric_card, create_status_indicator

def set_page_config():
    """Configura la página de Streamlit."""
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Aplicar estilos personalizados
    apply_custom_styling()

def display_header():
    """Muestra el header de la aplicación con información mejorada."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2E5BBA, #4A90B8); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
                box-shadow: 0 8px 24px rgba(46, 91, 186, 0.15);">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem; font-weight: 600;">
             Dermosan - Sistema de Diagnóstico Dermatológico
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.95;">
           Proyecto DERMOSAN – UNDC 2025
        </p>
        <div style="text-align: center; margin-top: 1.5rem;">
            <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.5rem 1.5rem; 
                         border-radius: 25px; font-size: 0.95rem; border: 1px solid rgba(255,255,255,0.3);">
                 IA ResNet152 |  95% Precisión |  10 Enfermedades
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_info():
    """Muestra información mejorada en la barra lateral."""
    with st.sidebar:
        st.markdown("### Información del Sistema")
        
        # Métricas del modelo en cajas coloridas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #27AE60, #2ECC71); color: white; 
                        padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 0.5rem;
                        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.2);">
                <h3 style="margin: 0; font-size: 1.5rem; font-weight: 600;">95%</h3>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Precisión</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3498DB, #5DADE2); color: white; 
                        padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 0.5rem;
                        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);">
                <h3 style="margin: 0; font-size: 1.5rem; font-weight: 600;">10</h3>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Enfermedades</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **Modelo:** ResNet152 con Transfer Learning  
        **Dataset:** +27,000 imágenes  
        **Input:** 224x224 RGB  
        **Tiempo promedio:** < 3 segundos  
        """)
        
        # Agregar estado del sistema
        st.markdown("### 🔋 Estado del Sistema")
        st.markdown(create_status_indicator("active", "Sistema Operativo"), unsafe_allow_html=True)
        st.markdown(create_status_indicator("active", "Modelo Cargado"), unsafe_allow_html=True)
        st.markdown(create_status_indicator("active", "IA Lista para Diagnóstico"), unsafe_allow_html=True)
        
        st.markdown("### Estadísticas del Dataset")
    
        dataset_stats = {
            "Eczema": 1677,
            "Melanoma": 3140,
            "Atopic Dermatitis": 1257,
            "Basal Cell Carcinoma (BCC)": 3323,
            "Melanocytic Nevi (NV)": 7970,
            "Benign Keratosis-like Lesions (BKL)": 2624,
            "Psoriasis / Lichen Planus & related diseases": 2055,
            "Seborrheic Keratoses & other benign tumors": 1847,
            "Tinea / Candidiasis & other fungal infections": 1702,
            "Warts / Molluscum & other viral infections": 2103
        }

        
        df_stats = pd.DataFrame(
            list(dataset_stats.items()), 
            columns=['Enfermedad', 'Imágenes']
        )
        st.dataframe(df_stats, hide_index=True, width='stretch')

def create_confidence_gauge(confidence: float) -> go.Figure:
    """
    Crea un gauge chart para mostrar la confianza con diseño médico.
    
    Args:
        confidence: Valor de confianza (0-1)
        
    Returns:
        Figura de Plotly
    """
    confidence_percentage = confidence * 100
    
    # Determinar color basado en confianza
    if confidence >= 0.8:
        color = "#27AE60"  # Verde médico
        bar_color = "rgba(39, 174, 96, 0.9)"
    elif confidence >= 0.6:
        color = "#F39C12"  # Naranja médico
        bar_color = "rgba(243, 156, 18, 0.9)"
    else:
        color = "#E74C3C"  # Rojo médico
        bar_color = "rgba(231, 76, 60, 0.9)"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_percentage,
        title={'text': "Nivel de Confianza (%)", 'font': {'size': 18, 'color': '#2E5BBA'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 40, 'color': '#2E5BBA', 'family': 'Arial Black'}},
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2, 
                'tickcolor': "#34495E",
                'tickfont': {'size': 14, 'color': '#34495E'}
            },
            'bar': {
                'color': bar_color, 
                'thickness': 0.8,
                'line': {'color': '#2E5BBA', 'width': 3}
            },
            'bgcolor': "rgba(240, 248, 255, 0.3)",
            'borderwidth': 4,
            'bordercolor': "#2E5BBA",
            'steps': [
                {'range': [0, 40], 'color': "rgba(231, 76, 60, 0.2)", 'name': 'Bajo'},
                {'range': [40, 60], 'color': "rgba(243, 156, 18, 0.2)", 'name': 'Medio'},
                {'range': [60, 80], 'color': "rgba(52, 152, 219, 0.2)", 'name': 'Bueno'},
                {'range': [80, 100], 'color': "rgba(39, 174, 96, 0.2)", 'name': 'Excelente'}
            ],
            'threshold': {
                'line': {'color': "#E74C3C", 'width': 5},
                'thickness': 0.9,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#34495E"
        )
    )
    return fig

def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Crea un gráfico de barras horizontal elegante con las probabilidades.
    
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
        fig = go.Figure()
        fig.update_layout(title="No hay datos para mostrar")
        return fig
    
    # Obtener colores y ordenar por probabilidad
    disease_data = [(disease, prob, DISEASE_INFO.get(disease, {}).get("color", "#636EFA")) 
                    for disease, prob in zip(diseases, probs)]
    disease_data.sort(key=lambda x: x[1], reverse=True)
    
    diseases_sorted, probs_sorted, colors_sorted = zip(*disease_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs_sorted),
            y=list(diseases_sorted),
            orientation='h',
            marker=dict(
                color=list(colors_sorted),
                line=dict(color='#2E5BBA', width=2),
                opacity=0.8
            ),
            text=[f"{prob:.1f}%" for prob in probs_sorted],
            textposition='outside',
            textfont=dict(size=12, color='#2E5BBA', family='Arial Bold')
        )
    ])
    
    # Calcular rango del eje X
    max_prob = max(probs_sorted) if probs_sorted else 100
    x_range = [0, max_prob * 1.15]
    
    fig.update_layout(
        title=dict(
            text="🔬 Análisis de Probabilidades por Enfermedad",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        xaxis_title="Probabilidad (%)",
        yaxis_title="Enfermedad Detectada",
        height=500,
        margin=dict(l=150, r=50, t=80, b=50),
        xaxis=dict(
            range=x_range,
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color='#34495E'
        ),
        xaxis_title_font=dict(size=14, color='#2E5BBA'),
        yaxis_title_font=dict(size=14, color='#2E5BBA'),
        showlegend=False
    )
    return fig

def create_compact_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Crea un gráfico de barras horizontal compacto para el panel principal.
    
    Args:
        probabilities: Diccionario con probabilidades por clase
        
    Returns:
        Figura de Plotly compacta
    """
    # Preparar datos - mostrar solo top 5
    diseases = list(probabilities.keys())
    probs = [probabilities[disease] * 100 for disease in diseases]
    
    if not probs:
        fig = go.Figure()
        fig.update_layout(title="No hay datos para mostrar")
        return fig
    
    # Obtener colores y ordenar por probabilidad (top 5)
    disease_data = [(disease, prob, DISEASE_INFO.get(disease, {}).get("color", "#636EFA")) 
                    for disease, prob in zip(diseases, probs)]
    disease_data.sort(key=lambda x: x[1], reverse=True)
    disease_data = disease_data[:5]  # Solo top 5
    
    diseases_sorted, probs_sorted, colors_sorted = zip(*disease_data)
    
    # Acortar nombres de enfermedades para vista compacta
    diseases_short = []
    for disease in diseases_sorted:
        if len(disease) > 20:
            diseases_short.append(disease[:17] + "...")
        else:
            diseases_short.append(disease)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs_sorted),
            y=list(diseases_short),
            orientation='h',
            marker=dict(
                color=list(colors_sorted),
                line=dict(color='#2E5BBA', width=1),
                opacity=0.8
            ),
            text=[f"{prob:.1f}%" for prob in probs_sorted],
            textposition='outside',
            textfont=dict(size=10, color='#2E5BBA', family='Arial Bold')
        )
    ])
    
    max_prob = max(probs_sorted) if probs_sorted else 100
    x_range = [0, max_prob * 1.2]
    
    fig.update_layout(
        height=300,  # Más compacto
        margin=dict(l=100, r=20, t=20, b=20),  # Márgenes reducidos
        xaxis=dict(
            range=x_range,
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=True,
            showticklabels=True
        ),
        yaxis=dict(
            gridcolor='rgba(46, 91, 186, 0.1)',
            showgrid=False,
            showticklabels=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, sans-serif",
            size=10,
            color='#34495E'
        ),
        showlegend=False
    )
    return fig

def display_results_container(content_func):
    """
    Wrapper para mostrar contenido en un contenedor con diseño mejorado.
    
    Args:
        content_func: Función que genera el contenido
    """
    st.markdown("""
    <div class="result-container">
    """, unsafe_allow_html=True)
    
    content_func()
    
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

def display_confidence_level(confidence: float):
    """
    Muestra el nivel de confianza con colores y estilo médico.
    
    Args:
        confidence: Valor de confianza (0-1)
    """
    confidence_percentage = confidence * 100
    
    if confidence >= 0.8:
        css_class = "confidence-high"
        level = "Alta"
        icon = "🟢"
    elif confidence >= 0.6:
        css_class = "confidence-medium" 
        level = "Media"
        icon = "🟡"
    else:
        css_class = "confidence-low"
        level = "Baja"
        icon = "🔴"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; 
                border: 2px solid #E8F4FD; margin: 1rem 0;">
        <h3 style="margin: 0; color: #2E5BBA;">Nivel de Confianza</h3>
        <div style="margin: 1rem 0;">
            <span style="font-size: 2rem;">{icon}</span>
            <span class="{css_class}" style="font-size: 2.5rem; margin-left: 0.5rem;">
                {confidence_percentage:.1f}%
            </span>
        </div>
        <p style="margin: 0; color: #7F8C8D;">Confianza: {level}</p>
    </div>
    """, unsafe_allow_html=True)

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
    Clínica: DERMOSAN – UNDC 2025
    
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
            <h2 style="margin: 0; font-size: 2rem;">95%</h2>
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
    st.dataframe(tech_df, hide_index=True, width='stretch')

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

def create_risk_assessment_chart(predicted_disease: str, confidence: float) -> go.Figure:
    """
    Crea un gráfico de evaluación de riesgo médico.
    
    Args:
        predicted_disease: Enfermedad predicha
        confidence: Nivel de confianza
        
    Returns:
        Figura de Plotly con evaluación de riesgo
    """
    # Obtener información de la enfermedad
    disease_info = DISEASE_INFO.get(predicted_disease, {})
    severity_level = disease_info.get("severity", "Moderada")
    
    # Definir niveles de riesgo
    risk_levels = {
        "Bajo": {"value": 25, "color": "#27AE60", "description": "Monitoreo regular"},
        "Moderado": {"value": 50, "color": "#F39C12", "description": "Consulta dermatológica"},
        "Alto": {"value": 75, "color": "#E67E22", "description": "Atención inmediata"},
        "Crítico": {"value": 100, "color": "#E74C3C", "description": "Urgencia médica"}
    }
    
    # Determinar riesgo basado en la enfermedad y confianza
    if predicted_disease in ["Melanoma", "Basal Cell Carcinoma (BCC)"]:
        if confidence >= 0.8:
            current_risk = "Alto"
        else:
            current_risk = "Moderado"
    elif predicted_disease in ["Atopic Dermatitis", "Eczema", "Psoriasis pictures Lichen Planus and related diseases"]:
        current_risk = "Moderado" if confidence >= 0.7 else "Bajo"
    else:
        current_risk = "Bajo" if confidence >= 0.8 else "Moderado"
    
    # Crear el gráfico
    categories = list(risk_levels.keys())
    values = [risk_levels[cat]["value"] for cat in categories]
    colors = [risk_levels[cat]["color"] for cat in categories]
    
    # Destacar el nivel actual
    opacity = [0.9 if cat == current_risk else 0.3 for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=colors,
                opacity=opacity,
                line=dict(color='#2E5BBA', width=2)
            ),
            text=[f"{risk_levels[cat]['description']}" for cat in categories],
            textposition='outside',
            textfont=dict(size=11, color='#2E5BBA')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"Evaluación de Riesgo: {current_risk.upper()}",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        xaxis_title="Nivel de Riesgo",
        yaxis_title="Urgencia (%)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color='#34495E'),
        xaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)'),
        yaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)'),
        showlegend=False
    )
    
    # Agregar línea indicadora del nivel actual
    current_value = risk_levels[current_risk]["value"]
    fig.add_hline(
        y=current_value,
        line_dash="dash",
        line_color=risk_levels[current_risk]["color"],
        line_width=3,
        annotation_text=f"Nivel Actual: {current_risk}",
        annotation_position="top right"
    )
    
    return fig

def create_comparison_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Crea un gráfico de comparación entre las top 3 enfermedades más probables.
    
    Args:
        probabilities: Diccionario con probabilidades por clase
        
    Returns:
        Figura de Plotly
    """
    # Obtener top 3 enfermedades
    sorted_diseases = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if len(sorted_diseases) < 2:
        # Gráfico vacío si no hay suficientes datos
        fig = go.Figure()
        fig.update_layout(title="Datos insuficientes para comparación")
        return fig
    
    diseases, probs = zip(*sorted_diseases)
    probs_percent = [p * 100 for p in probs]
    colors = [DISEASE_INFO.get(disease, {}).get("color", "#636EFA") for disease in diseases]
    
    # Crear gráfico de dona (pie chart)
    fig = go.Figure(data=[
        go.Pie(
            labels=diseases,
            values=probs_percent,
            hole=0.4,
            marker=dict(
                colors=colors,
                line=dict(color='#2E5BBA', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='white', family='Arial Bold'),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Comparación Top 3 Diagnósticos",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color='#34495E'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    # Agregar anotación en el centro
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>{diseases[0]}</b><br>{probs_percent[0]:.1f}%",
        showarrow=False,
        font=dict(size=16, color='#2E5BBA', family='Arial Bold'),
        align="center"
    )
    
    return fig

def create_severity_timeline() -> go.Figure:
    """
    Crea un gráfico de línea temporal mostrando la evolución típica de severidad.
    
    Returns:
        Figura de Plotly
    """
    # Datos simulados de evolución temporal
    timeline_data = {
        "Días": [0, 7, 14, 21, 30, 60, 90],
        "Sin Tratamiento": [30, 45, 60, 75, 85, 90, 95],
        "Con Tratamiento": [30, 25, 20, 15, 10, 5, 3],
        "Tratamiento Tardío": [30, 50, 65, 55, 40, 25, 15]
    }
    
    fig = go.Figure()
    
    # Línea sin tratamiento
    fig.add_trace(go.Scatter(
        x=timeline_data["Días"],
        y=timeline_data["Sin Tratamiento"],
        mode='lines+markers',
        name='Sin Tratamiento',
        line=dict(color='#E74C3C', width=3, dash='solid'),
        marker=dict(size=8, color='#E74C3C')
    ))
    
    # Línea con tratamiento temprano
    fig.add_trace(go.Scatter(
        x=timeline_data["Días"],
        y=timeline_data["Con Tratamiento"],
        mode='lines+markers',
        name='Tratamiento Temprano',
        line=dict(color='#27AE60', width=3, dash='solid'),
        marker=dict(size=8, color='#27AE60')
    ))
    
    # Línea con tratamiento tardío
    fig.add_trace(go.Scatter(
        x=timeline_data["Días"],
        y=timeline_data["Tratamiento Tardío"],
        mode='lines+markers',
        name='Tratamiento Tardío',
        line=dict(color='#F39C12', width=3, dash='dash'),
        marker=dict(size=8, color='#F39C12')
    ))
    
    fig.update_layout(
        title=dict(
            text="Evolución Temporal de Severidad",
            font=dict(size=18, color='#2E5BBA', family="Arial Bold"),
            x=0.5
        ),
        xaxis_title="Tiempo (días)",
        yaxis_title="Severidad (%)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color='#34495E'),
        xaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(46, 91, 186, 0.1)', showgrid=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def display_medical_footer():
    """Muestra un footer médico profesional."""
    st.markdown("""
    <div class="footer">
        <h4 style="margin: 0 0 1rem 0;"> Dermosan - Sistema de Diagnóstico Dermatológico</h4>
        <p style="margin: 0.5rem 0; opacity: 0.9;">
            Proyecto DERMOSAN – UNDC 2025 | Desarrollado con IA ResNet152
        </p>
        <p style="margin: 0.5rem 0; font-size: 0.9rem; opacity: 0.8;">
            🔬 Este sistema es una herramienta de apoyo diagnóstico. 
            Siempre consulte con un dermatólogo profesional.
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <small style="opacity: 0.7;">
                © 2025 Dermosan | Precisión: 95% | 10 Enfermedades Detectables
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)
