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

from src.config import MODEL_PATH, IMG_SIZE, DISEASE_CLASSES, CONFIDENCE_THRESHOLDS

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
            if os.path.exists(MODEL_PATH):
                # Intentar cargar con configuraciones compatibles
                try:
                    # Método 1: Carga normal
                    self.model = tf.keras.models.load_model(MODEL_PATH)
                    logging.info(f"Modelo cargado exitosamente desde {MODEL_PATH}")
                except Exception as e1:
                    logging.warning(f"Fallo método 1: {str(e1)}")
                    try:
                        # Método 2: Cargar con compile=False
                        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                        # Recompilar el modelo manualmente
                        self.model.compile(
                            optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        logging.info(f"Modelo cargado con compile=False desde {MODEL_PATH}")
                    except Exception as e2:
                        logging.warning(f"Fallo método 2: {str(e2)}")
                        # Método 3: Cargar solo pesos y reconstruir arquitectura
                        self._load_model_weights_only()
            else:
                raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def _load_model_weights_only(self):
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
            self.model.load_weights(MODEL_PATH)
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
        
        # Calcular métricas básicas
        blur_score = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Evaluación de calidad
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
            "is_suitable": quality_score >= 75
        }
        
    except Exception as e:
        logging.error(f"Error en análisis de calidad: {str(e)}")
        return {
            "quality_score": 0,
            "issues": ["Error al analizar la imagen"],
            "is_suitable": False
        }
