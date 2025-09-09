"""
Módulo de predicción para el sistema de diagnóstico dermatológico
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import cv2
from typing import Dict, Tuple, List
import logging

from src.config import MODEL_PATH, MODEL_PATH_FALLBACK, IMG_SIZE, DISEASE_CLASSES, CONFIDENCE_THRESHOLDS

class DermatologyPredictor:
    """
    Clase para realizar predicciones de enfermedades dermatológicas
    usando el modelo ResNet152 entrenado.
    """
    
    def __init__(self):
        """Inicializa el predictor cargando el modelo."""
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Carga el modelo entrenado."""
        try:
            # Intentar primera ruta (corregida)
            if os.path.exists(MODEL_PATH):
                model_path = MODEL_PATH
            # Intentar ruta de respaldo
            elif os.path.exists(MODEL_PATH_FALLBACK):
                model_path = MODEL_PATH_FALLBACK
                logging.warning(f"Usando ruta de respaldo: {MODEL_PATH_FALLBACK}")
            else:
                raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH} ni en {MODEL_PATH_FALLBACK}")
            
            # Intentar cargar con configuraciones compatibles
            try:
                # Método 1: Carga normal
                self.model = tf.keras.models.load_model(model_path)
                logging.info(f"Modelo cargado exitosamente desde {model_path}")
            except Exception as e1:
                logging.warning(f"Fallo método 1: {str(e1)}")
                try:
                    # Método 2: Cargar con compile=False
                    self.model = tf.keras.models.load_model(model_path, compile=False)
                    # Recompilar el modelo manualmente
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    logging.info(f"Modelo cargado con compile=False desde {model_path}")
                except Exception as e2:
                    logging.warning(f"Fallo método 2: {str(e2)}")
                    # Método 3: Cargar solo pesos y reconstruir arquitectura
                    self._load_model_weights_only(model_path)
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def _load_model_weights_only(self, model_path):
        """Método alternativo: reconstruir modelo y cargar solo pesos."""
        try:
            from tensorflow.keras.applications import ResNet152
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
            from tensorflow.keras.models import Model
            
            logging.info("Intentando reconstruir arquitectura del modelo...")
            
            # Reconstruir la arquitectura del modelo
            base = ResNet152(
                weights='imagenet',
                include_top=False,
                input_shape=(*IMG_SIZE, 3)
            )
            
            # Hacer las últimas 50 capas entrenables (como en el script original)
            for layer in base.layers[:-50]:
                layer.trainable = False
            
            # Añadir capas de clasificación
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.35)(x)
            x = Dense(256, activation='relu')(x)
            out = Dense(len(DISEASE_CLASSES), activation='softmax')(x)
            
            self.model = Model(inputs=base.input, outputs=out)
            
            # Compilar
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Cargar solo los pesos
            self.model.load_weights(model_path)
            logging.info("Modelo reconstruido y pesos cargados exitosamente")
            
        except Exception as e:
            logging.error(f"Error al reconstruir modelo: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesa la imagen para el modelo.
        
        Args:
            image: Imagen PIL
            
        Returns:
            Array numpy preprocessado
        """
        try:
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionar
            image = image.resize(IMG_SIZE)
            
            # Convertir a array numpy
            img_array = np.array(image)
            
            # Aplicar preprocesamiento de ResNet
            img_array = preprocess_input(img_array)
            
            # Añadir dimensión batch
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logging.error(f"Error en preprocesamiento: {str(e)}")
            raise
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Realiza predicción sobre una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            Diccionario con resultados de predicción
        """
        try:
            if self.model is None:
                raise ValueError("Modelo no cargado")
            
            # Preprocesar imagen
            processed_image = self.preprocess_image(image)
            
            # Realizar predicción
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Obtener clase predicha
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = DISEASE_CLASSES[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Determinar nivel de confianza
            confidence_level = self._get_confidence_level(confidence)
            
            # Obtener top 3 predicciones
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                {
                    "disease": DISEASE_CLASSES[idx],
                    "probability": float(probabilities[idx]),
                    "percentage": f"{probabilities[idx]*100:.1f}%"
                }
                for idx in top_3_indices
            ]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "confidence_percentage": f"{confidence*100:.1f}%",
                "top_3_predictions": top_3_predictions,
                "all_probabilities": {
                    DISEASE_CLASSES[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            logging.error(f"Error en predicción: {str(e)}")
            raise
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Determina el nivel de confianza basado en el threshold.
        
        Args:
            confidence: Valor de confianza
            
        Returns:
            Nivel de confianza como string
        """
        if confidence >= CONFIDENCE_THRESHOLDS["high"]:
            return "Alta"
        elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
            return "Media"
        elif confidence >= CONFIDENCE_THRESHOLDS["low"]:
            return "Baja"
        else:
            return "Muy Baja"
    
    def get_medical_recommendation(self, prediction_result: Dict) -> Dict:
        """
        Genera recomendaciones médicas basadas en la predicción.
        
        Args:
            prediction_result: Resultado de predicción
            
        Returns:
            Diccionario con recomendaciones
        """
        predicted_class = prediction_result["predicted_class"]
        confidence_level = prediction_result["confidence_level"]
        
        # Recomendaciones basadas en confianza
        if confidence_level == "Alta":
            urgency = "Consulta recomendada"
            action = "Programar cita con dermatólogo para confirmación"
        elif confidence_level == "Media":
            urgency = "Evaluación adicional necesaria"
            action = "Se recomienda segunda opinión y posible biopsia"
        else:
            urgency = "Diagnóstico incierto"
            action = "Requiere evaluación clínica presencial inmediata"
        
        # Recomendaciones específicas para condiciones graves
        if "Melanoma" in predicted_class and confidence_level in ["Alta", "Media"]:
            urgency = "URGENTE - Atención inmediata"
            action = "Derivar a oncólogo dermatológico de inmediato"
        elif "Carcinoma" in predicted_class and confidence_level in ["Alta", "Media"]:
            urgency = "Prioritario"
            action = "Programar biopsia y evaluación oncológica"
        
        return {
            "urgency": urgency,
            "recommended_action": action,
            "follow_up": "Seguimiento en 2-4 semanas según evolución"
        }

def analyze_image_quality(image: Image.Image) -> Dict:
    """
    Analiza la calidad de la imagen para diagnóstico.
    
    Args:
        image: Imagen PIL
        
    Returns:
        Diccionario con métricas de calidad
    """
    try:
        # Convertir a array numpy
        img_array = np.array(image)
        
        # 1. VALIDACIÓN CRÍTICA: Verificar que es una imagen médica válida
        validation_result = validate_medical_image(img_array)
        if not validation_result["is_valid"]:
            return {
                "quality_score": 0,
                "blur_score": 0,
                "brightness": 0,
                "contrast": 0,
                "resolution": f"{image.size[0]}x{image.size[1]}",
                "issues": validation_result["issues"],
                "is_suitable": False,
                "is_medical_image": False,
                "validation_error": validation_result["error_type"]
            }
        
        # 2. Calcular métricas básicas de calidad
        blur_score = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # 3. Evaluación de calidad
        quality_score = 0
        issues = []
        
        # Evaluar nitidez
        if blur_score < 100:
            issues.append("Imagen borrosa - considere tomar nueva foto")
        else:
            quality_score += 25
            
        # Evaluar brillo
        if brightness < 50:
            issues.append("Imagen muy oscura")
        elif brightness > 200:
            issues.append("Imagen muy brillante")
        else:
            quality_score += 25
            
        # Evaluar contraste
        if contrast < 30:
            issues.append("Bajo contraste")
        else:
            quality_score += 25
            
        # Evaluar resolución
        width, height = image.size
        if width < 224 or height < 224:
            issues.append("Resolución muy baja")
        else:
            quality_score += 25
            
        return {
            "quality_score": quality_score,
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "resolution": f"{width}x{height}",
            "issues": issues,
            "is_suitable": quality_score >= 75,
            "is_medical_image": True,
            "validation_error": None
        }
        
    except Exception as e:
        logging.error(f"Error en análisis de calidad: {str(e)}")
        return {
            "quality_score": 0,
            "issues": ["Error al analizar la imagen"],
            "is_suitable": False,
            "is_medical_image": False,
            "validation_error": "processing_error"
        }

def validate_medical_image(img_array: np.ndarray) -> Dict:
    """
    Valida que la imagen sea apropiada para análisis dermatológico.
    
    Args:
        img_array: Array numpy de la imagen
        
    Returns:
        Diccionario con resultado de validación
    """
    try:
        # 1. Verificar que no sea texto/documento
        if is_text_document(img_array):
            return {
                "is_valid": False,
                "error_type": "text_document",
                "issues": [
                    "🚫 IMAGEN NO VÁLIDA: Se detectó un documento de texto",
                    "📋 Este sistema solo analiza fotografías de piel/lesiones cutáneas",
                    "📸 Por favor, suba una imagen dermatológica real"
                ]
            }
        
        # 2. Verificar que tenga características de imagen médica
        if not has_skin_characteristics(img_array):
            return {
                "is_valid": False,
                "error_type": "not_medical",
                "issues": [
                    "🚫 IMAGEN NO MÉDICA: No se detectaron características de piel",
                    "🏥 Este sistema está diseñado para imágenes dermatológicas",
                    "📸 Suba una fotografía clara de piel o lesión cutánea"
                ]
            }
        
        # 3. Verificar que no sea una captura de pantalla
        if is_screenshot(img_array):
            return {
                "is_valid": False,
                "error_type": "screenshot",
                "issues": [
                    "🚫 CAPTURA DE PANTALLA DETECTADA",
                    "📱 No se permiten capturas de pantalla o imágenes de documentos",
                    "📸 Use una cámara para fotografiar directamente la piel"
                ]
            }
        
        return {
            "is_valid": True,
            "error_type": None,
            "issues": []
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "error_type": "validation_error",
            "issues": [f"Error en validación: {str(e)}"]
        }

def is_text_document(img_array: np.ndarray) -> bool:
    """Detecta si la imagen es un documento de texto."""
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Detectar mucho texto (áreas blancas/negras bien definidas)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        white_ratio = np.sum(binary == 255) / binary.size
        
        # 2. Detectar patrones de texto (líneas horizontales largas)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Más restrictivo
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_ratio = np.sum(horizontal_lines > 0) / horizontal_lines.size
        
        # 3. Detectar texto real usando detección de caracteres
        # Buscar regiones rectangulares pequeñas que podrían ser letras
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_like_regions = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Tamaño típico de caracteres
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 3:  # Proporción típica de letras
                    text_like_regions += 1
        
        text_density = text_like_regions / (gray.shape[0] * gray.shape[1] / 10000)  # Normalizar por área
        
        # 4. Detectar bordes muy rectos y organizados (típico de documentos)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Criterios MÁS ESTRICTOS para documento de texto
        # Solo considerar documento si MÚLTIPLES indicadores están presentes
        strong_indicators = 0
        
        if white_ratio > 0.85:  # Fondo muy blanco (más restrictivo)
            strong_indicators += 1
        if horizontal_ratio > 0.15:  # Muchas líneas horizontales largas
            strong_indicators += 1  
        if text_density > 5:  # Densidad alta de regiones tipo texto
            strong_indicators += 1
        if edge_ratio > 0.2 and white_ratio > 0.7:  # Muchos bordes + fondo blanco
            strong_indicators += 1
            
        # Necesita al menos 2 indicadores fuertes para ser considerado documento
        is_document = strong_indicators >= 2
        
        return is_document
        
    except:
        return False

def has_skin_characteristics(img_array: np.ndarray) -> bool:
    """Verifica si la imagen tiene características típicas de piel."""
    try:
        # Convertir a HSV para análisis de color
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 1. Rangos de color típicos de piel humana (más amplios)
        skin_ranges = [
            # Piel muy clara
            ([0, 10, 60], [25, 255, 255]),
            # Piel clara a media
            ([0, 15, 30], [30, 255, 255]),
            # Piel media a oscura
            ([5, 25, 20], [25, 255, 200]),
            # Piel con inflamación/enrojecimiento
            ([0, 30, 80], [15, 255, 255])
        ]
        
        skin_pixels = 0
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        for lower, upper in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            skin_pixels += np.sum(mask > 0)
        
        skin_ratio = skin_pixels / total_pixels
        
        # 2. Verificar textura orgánica vs geométrica
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detectar líneas muy rectas y largas (no deseadas en piel)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)  # Threshold más alto
        straight_lines = 0
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # Solo contar líneas muy horizontales o verticales (documentos/UI)
                if abs(theta) < 0.1 or abs(theta - np.pi/2) < 0.1 or abs(theta - np.pi) < 0.1:
                    straight_lines += 1
        
        has_many_straight_lines = straight_lines > 15  # Más tolerante
        
        # 3. Verificar que no sea una imagen muy uniforme (típico de capturas)
        std_dev = np.std(gray)
        is_too_uniform = std_dev < 15
        
        # La imagen tiene características de piel si:
        # - Tiene tonos de piel O
        # - No tiene muchas líneas rectas perfectas Y no es muy uniforme
        has_skin_tones = skin_ratio > 0.05  # Más tolerante
        not_geometric = not has_many_straight_lines and not is_too_uniform
        
        return has_skin_tones or not_geometric
        
    except:
        return True  # En caso de error, permitir análisis

def is_screenshot(img_array: np.ndarray) -> bool:
    """Detecta si es una captura de pantalla de interfaz."""
    try:
        # 1. Detectar bordes muy definidos (típico de UI)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 2. Detectar rectángulos perfectos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perfect_rects = 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Solo contornos grandes
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Es un rectángulo
                    perfect_rects += 1
        
        # 3. Verificar colores típicos de UI (mucho blanco/gris)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        
        # Es screenshot si tiene muchos rectángulos perfectos y pocos colores únicos
        return perfect_rects > 5 and unique_colors < 50
        
    except:
        return False
