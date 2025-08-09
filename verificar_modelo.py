"""
Script específico para probar la carga del modelo Dermosan
"""

import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

# Agregar directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verificar_archivo_modelo():
    """Verifica que el archivo del modelo existe y sus propiedades."""
    model_path = "Modelo Entrenado/best_resnet152.h5"
    
    print("🔍 Verificando archivo del modelo...")
    print(f"Ruta: {model_path}")
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Archivo encontrado - Tamaño: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ Archivo no encontrado en: {os.path.abspath(model_path)}")
        return False

def probar_metodos_carga():
    """Prueba diferentes métodos de carga del modelo."""
    model_path = "Modelo Entrenado/best_resnet152.h5"
    
    print("\n🧪 Probando métodos de carga del modelo...")
    
    # Método 1: Carga normal
    print("\n[Método 1] Carga normal...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Método 1 exitoso")
        print(f"   - Entrada: {model.input_shape}")
        print(f"   - Salida: {model.output_shape}")
        print(f"   - Parámetros: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Método 1 falló: {str(e)}")
    
    # Método 2: Carga sin compilar
    print("\n[Método 2] Carga sin compilar...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Método 2 exitoso")
        
        # Recompilar
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modelo recompilado")
        print(f"   - Entrada: {model.input_shape}")
        print(f"   - Salida: {model.output_shape}")
        print(f"   - Parámetros: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Método 2 falló: {str(e)}")
    
    # Método 3: Reconstruir arquitectura
    print("\n[Método 3] Reconstruir arquitectura...")
    try:
        model = reconstruir_modelo_y_cargar_pesos(model_path)
        print("✅ Método 3 exitoso")
        return model
    except Exception as e:
        print(f"❌ Método 3 falló: {str(e)}")
    
    return None

def reconstruir_modelo_y_cargar_pesos(model_path):
    """Reconstruye la arquitectura del modelo y carga los pesos."""
    from tensorflow.keras.applications import ResNet152
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    
    print("   - Creando base ResNet152...")
    base = ResNet152(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    print("   - Configurando capas entrenables...")
    for layer in base.layers[:-50]:
        layer.trainable = False
    
    print("   - Añadiendo capas de clasificación...")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)  # 10 clases
    
    model = Model(inputs=base.input, outputs=out)
    
    print("   - Compilando modelo...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("   - Cargando pesos...")
    model.load_weights(model_path)
    
    return model

def probar_prediccion(model):
    """Prueba una predicción simple con el modelo."""
    print("\n🎯 Probando predicción...")
    
    try:
        # Crear imagen de prueba
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_image = (test_image - 0.5) * 2  # Normalizar como ResNet
        
        print("   - Realizando predicción...")
        predictions = model.predict(test_image, verbose=0)
        
        print(f"✅ Predicción exitosa")
        print(f"   - Shape salida: {predictions.shape}")
        print(f"   - Suma probabilidades: {predictions.sum():.3f}")
        print(f"   - Clase predicha: {np.argmax(predictions[0])}")
        print(f"   - Confianza máxima: {np.max(predictions[0]):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        return False

def main():
    print("🏥 DERMOSAN - VERIFICADOR DE MODELO")
    print("=" * 50)
    
    # Verificar archivo
    if not verificar_archivo_modelo():
        print("\n❌ No se puede continuar sin el archivo del modelo")
        return
    
    # Probar carga
    model = probar_metodos_carga()
    
    if model is None:
        print("\n❌ No se pudo cargar el modelo con ningún método")
        return
    
    # Probar predicción
    if probar_prediccion(model):
        print("\n🎉 ¡MODELO VERIFICADO EXITOSAMENTE!")
        print("\nEl modelo está listo para usar en la aplicación Streamlit")
    else:
        print("\n⚠️ Modelo cargado pero con problemas en predicción")
    
    print("\n" + "=" * 50)
    print("Para ejecutar la aplicación: streamlit run app.py")

if __name__ == "__main__":
    main()
