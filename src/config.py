"""
Configuración del sistema de diagnóstico dermatológico
"""

import os
from typing import Dict, List, Tuple

# ── Configuración del modelo ─────────────────────────────────────────────────
MODEL_PATH = "models/best_resnet152.h5"  # Ruta corregida
MODEL_PATH_FALLBACK = "Modelo Entrenado/best_resnet152.h5"  # Ruta alternativa
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ── Clases de enfermedades dermatológicas ────────────────────────────────────
DISEASE_CLASSES = {
    0: "Eczema",
    1: "Warts Molluscum and Viral Infections", 
    2: "Melanoma",
    3: "Atopic Dermatitis",
    4: "Basal Cell Carcinoma (BCC)",
    5: "Melanocytic Nevi (NV)",
    6: "Benign Keratosis-like Lesions (BKL)",
    7: "Psoriasis Lichen Planus",
    8: "Seborrheic Keratoses",
    9: "Tinea Ringworm Candidiasis"
}

# ── Información médica detallada ──────────────────────────────────────────────
DISEASE_INFO = {
    "Eczema": {
        "description": "Inflamación crónica de la piel caracterizada por erupciones rojas, picazón y descamación.",
        "severity": "Leve a Moderada",
        "treatment": "Hidratantes, corticosteroides tópicos, antihistamínicos",
        "color": "#E74C3C"  # Rojo médico
    },
    "Warts Molluscum and Viral Infections": {
        "description": "Infecciones virales de la piel que causan pequeñas protuberancias o verrugas.",
        "severity": "Leve",
        "treatment": "Crioterapia, medicamentos tópicos, observación",
        "color": "#1ABC9C"  # Verde azulado médico
    },
    "Melanoma": {
        "description": "Tipo más peligroso de cáncer de piel que se desarrolla en los melanocitos.",
        "severity": "Grave - Requiere atención inmediata",
        "treatment": "Cirugía, inmunoterapia, terapia dirigida",
        "color": "#8E44AD"  # Púrpura médico (crítico)
    },
    "Atopic Dermatitis": {
        "description": "Forma de eczema crónico común en niños, caracterizado por piel seca y con picazón.",
        "severity": "Leve a Moderada",
        "treatment": "Hidratantes, corticosteroides, inmunomoduladores",
        "color": "#F39C12"  # Naranja médico
    },
    "Basal Cell Carcinoma (BCC)": {
        "description": "Tipo más común de cáncer de piel, crecimiento lento y raramente metastásico.",
        "severity": "Moderada",
        "treatment": "Cirugía, crioterapia, medicamentos tópicos",
        "color": "#C0392B"  # Rojo oscuro médico
    },
    "Melanocytic Nevi (NV)": {
        "description": "Lunares benignos comunes, generalmente no requieren tratamiento.",
        "severity": "Benigna",
        "treatment": "Observación, biopsia si hay cambios",
        "color": "#27AE60"  # Verde médico
    },
    "Benign Keratosis-like Lesions (BKL)": {
        "description": "Lesiones benignas de la piel, incluye queratosis seborreica y lesiones similares.",
        "severity": "Benigna",
        "treatment": "Observación, remoción cosmética si se desea",
        "color": "#3498DB"  # Azul médico
    },
    "Psoriasis Lichen Planus": {
        "description": "Enfermedades inflamatorias crónicas de la piel con placas escamosas.",
        "severity": "Moderada",
        "treatment": "Corticosteroides, inmunosupresores, fototerapia",
        "color": "#9B59B6"  # Púrpura claro médico
    },
    "Seborrheic Keratoses": {
        "description": "Crecimientos benignos de la piel, comunes en adultos mayores.",
        "severity": "Benigna",
        "treatment": "Observación, remoción cosmética",
        "color": "#16A085"  # Verde oscuro médico
    },
    "Tinea Ringworm Candidiasis": {
        "description": "Infecciones fúngicas de la piel que causan erupciones circulares o irritación.",
        "severity": "Leve a Moderada",
        "treatment": "Antifúngicos tópicos u orales",
        "color": "#E67E22"  # Naranja oscuro médico
    }
}

# ── Configuración de la aplicación ────────────────────────────────────────────
APP_CONFIG = {
    "title": " Dermosan - Sistema de Diagnóstico Dermatológico",
    "subtitle": "Clínicas de San Vicente, Cañete",
    "description": "Sistema automatizado de diagnóstico de enfermedades dermatológicas usando Deep Learning"
}

# ── Configuración de confianza ────────────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,      # Alta confianza
    "medium": 0.6,    # Confianza media
    "low": 0.4        # Baja confianza
}
