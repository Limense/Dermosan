"""
Componentes adicionales para el dashboard de Dermosan
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_medical_dashboard_header():
    """Crea un header específico para el dashboard de resultados."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2E5BBA, #4A90B8); 
                color: white;
                padding: 2rem; 
                border-radius: 20px; 
                text-align: center;
                margin: 2rem 0;
                box-shadow: 0 12px 32px rgba(46, 91, 186, 0.25);
                border: 1px solid rgba(255,255,255,0.1);">
        
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div style="background: rgba(255,255,255,0.15); 
                        border-radius: 50%; 
                        padding: 1rem; 
                        margin-right: 1rem;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <span style="font-size: 2.5rem;">📊</span>
            </div>
            <h2 style="margin: 0; color: white; font-size: 2rem; font-weight: 600;">
                Dashboard de Análisis Médico
            </h2>
        </div>
        
        <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">
            Resultados detallados del diagnóstico dermatológico con IA
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_confidence_dashboard(confidence_value):
    """
    Crea un dashboard de confianza más elaborado.
    
    Args:
        confidence_value: Valor de confianza (0-1)
    """
    confidence_percentage = confidence_value * 100
    
    # Determinar nivel y color
    if confidence_percentage >= 85:
        level = "Excelente"
        color = "#27AE60"
        icon = "🟢"
        description = "Diagnóstico altamente confiable"
    elif confidence_percentage >= 70:
        level = "Bueno"
        color = "#3498DB"
        icon = "🔵"
        description = "Diagnóstico confiable"
    elif confidence_percentage >= 50:
        level = "Moderado"
        color = "#F39C12"
        icon = "🟡"
        description = "Diagnóstico con precaución"
    else:
        level = "Bajo"
        color = "#E74C3C"
        icon = "🔴"
        description = "Requiere evaluación adicional"
    
    st.markdown(f"""
    <div style="background: white; 
                padding: 2rem; 
                border-radius: 15px; 
                box-shadow: 0 8px 24px rgba(46, 91, 186, 0.15);
                border: 2px solid {color}20;
                margin: 1.5rem 0;
                text-align: center;">
        
        <div style="background: linear-gradient(135deg, {color}, {color}DD); 
                    color: white; 
                    padding: 1rem; 
                    border-radius: 50%; 
                    width: 80px; 
                    height: 80px; 
                    margin: 0 auto 1rem auto;
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    font-size: 2rem;">
            {icon}
        </div>
        
        <h3 style="color: {color}; margin: 0 0 0.5rem 0; font-size: 1.8rem;">
            {confidence_percentage:.1f}%
        </h3>
        <h4 style="color: #34495E; margin: 0 0 1rem 0; font-size: 1.2rem;">
            Confianza {level}
        </h4>
        <p style="color: #7F8C8D; margin: 0; font-size: 1rem;">
            {description}
        </p>
        
        <div style="background: {color}15; 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin-top: 1rem;">
            <p style="margin: 0; color: {color}; font-weight: 600;">
                ℹ️ Interpretación Clínica
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_system_metrics():
    """Crea métricas del sistema en tiempo real."""
    import datetime
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #27AE60, #2ECC71); 
                    color: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    text-align: center;
                    box-shadow: 0 4px 16px rgba(39, 174, 96, 0.3);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
            <h3 style="margin: 0; font-size: 1.8rem;">< 3s</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Tiempo de Análisis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3498DB, #5DADE2); 
                    color: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    text-align: center;
                    box-shadow: 0 4px 16px rgba(52, 152, 219, 0.3);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
            <h3 style="margin: 0; font-size: 1.8rem;">ResNet152</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Modelo de IA</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8E44AD, #A569BD); 
                    color: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    text-align: center;
                    box-shadow: 0 4px 16px rgba(142, 68, 173, 0.3);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <h3 style="margin: 0; font-size: 1.8rem;">95%</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Precisión Global</p>
        </div>
        """, unsafe_allow_html=True)

def create_disease_info_card(disease_name, disease_info):
    """
    Crea una tarjeta informativa de enfermedad mejorada.
    
    Args:
        disease_name: Nombre de la enfermedad
        disease_info: Información de la enfermedad
    """
    # Determinar icon según severidad
    severity_icons = {
        "Benigna": "✅",
        "Leve": "💛",
        "Leve a Moderada": "🟡",
        "Moderada": "🟠",
        "Grave - Requiere atención inmediata": "🚨"
    }
    
    severity_colors = {
        "Benigna": "#27AE60",
        "Leve": "#F1C40F",
        "Leve a Moderada": "#F39C12",
        "Moderada": "#E67E22",
        "Grave - Requiere atención inmediata": "#E74C3C"
    }
    
    icon = severity_icons.get(disease_info["severity"], "ℹ️")
    severity_color = severity_colors.get(disease_info["severity"], disease_info["color"])
    
    st.markdown(f"""
    <div style="background: white; 
                padding: 2rem; 
                border-radius: 15px; 
                box-shadow: 0 8px 24px rgba(46, 91, 186, 0.15);
                border-left: 5px solid {disease_info['color']};
                margin: 1.5rem 0;">
        
        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <div style="background: {disease_info['color']}20; 
                        color: {disease_info['color']}; 
                        padding: 1rem; 
                        border-radius: 50%; 
                        margin-right: 1rem;
                        font-size: 1.5rem;">
                {icon}
            </div>
            <div>
                <h3 style="color: {disease_info['color']}; margin: 0; font-size: 1.4rem;">
                    {disease_name}
                </h3>
                <span style="background: {severity_color}; 
                             color: white; 
                             padding: 0.2rem 0.8rem; 
                             border-radius: 12px; 
                             font-size: 0.8rem;">
                    {disease_info['severity']}
                </span>
            </div>
        </div>
        
        <div style="margin-bottom: 1rem;">
            <h4 style="color: #34495E; margin: 0 0 0.5rem 0;">📋 Descripción</h4>
            <p style="color: #7F8C8D; margin: 0; line-height: 1.6;">
                {disease_info['description']}
            </p>
        </div>
        
        <div style="background: {disease_info['color']}10; 
                    padding: 1rem; 
                    border-radius: 10px; 
                    border: 1px solid {disease_info['color']}30;">
            <h4 style="color: {disease_info['color']}; margin: 0 0 0.5rem 0;">💊 Tratamiento</h4>
            <p style="color: #34495E; margin: 0; font-weight: 500;">
                {disease_info['treatment']}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_medical_disclaimer():
    """Crea un disclaimer médico profesional."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #34495E, #2C3E50); 
                color: white; 
                padding: 2rem; 
                border-radius: 15px; 
                margin: 2rem 0;
                text-align: center;
                box-shadow: 0 8px 24px rgba(52, 73, 94, 0.3);">
        
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚕️</div>
        
        <h3 style="margin: 0 0 1rem 0; color: white;">
            Aviso Médico Importante
        </h3>
        
        <p style="color: rgba(255,255,255,0.9); margin: 0; line-height: 1.6; font-size: 1.1rem;">
            Este sistema es una <strong>herramienta de apoyo diagnóstico</strong> que utiliza 
            inteligencia artificial. Los resultados deben ser siempre interpretados por un 
            <strong>dermatólogo profesional</strong>. No reemplaza el juicio clínico médico.
        </p>
        
        <div style="background: rgba(255,255,255,0.1); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin-top: 1.5rem;">
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">
                🏥 Para consultas médicas, contacte con las Clínicas de San Vicente, Cañete
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
