"""
Script de prueba para verificar que el sistema Dermosan funciona correctamente
"""

import os
import sys
import traceback
from PIL import Image
import numpy as np

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Prueba la carga del modelo."""
    print("🔍 Probando carga del modelo...")
    try:
        from src.predictor import DermatologyPredictor
        predictor = DermatologyPredictor()
        print("✅ Modelo cargado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {str(e)}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Prueba el procesamiento de imágenes."""
    print("\n🖼️ Probando procesamiento de imágenes...")
    try:
        from src.predictor import analyze_image_quality
        
        # Crear imagen de prueba
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Analizar calidad
        quality_result = analyze_image_quality(test_image)
        print(f"✅ Análisis de calidad: {quality_result['quality_score']}/100")
        return True
    except Exception as e:
        print(f"❌ Error en procesamiento de imágenes: {str(e)}")
        traceback.print_exc()
        return False

def test_prediction():
    """Prueba una predicción completa."""
    print("\n🎯 Probando predicción completa...")
    try:
        from src.predictor import DermatologyPredictor
        
        predictor = DermatologyPredictor()
        
        # Crear imagen de prueba
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # Realizar predicción
        result = predictor.predict(test_image)
        
        print(f"✅ Predicción exitosa:")
        print(f"   - Clase: {result['predicted_class']}")
        print(f"   - Confianza: {result['confidence_percentage']}")
        print(f"   - Nivel: {result['confidence_level']}")
        
        # Probar recomendaciones
        recommendations = predictor.get_medical_recommendation(result)
        print(f"   - Urgencia: {recommendations['urgency']}")
        
        return True
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Prueba los componentes de Streamlit."""
    print("\n🎨 Probando componentes de interfaz...")
    try:
        from src.utils import create_confidence_gauge, create_probability_chart
        from src.config import DISEASE_CLASSES
        
        # Probar gauge
        fig_gauge = create_confidence_gauge(0.85)
        print("✅ Gauge de confianza creado")
        
        # Probar gráfico de probabilidades
        test_probs = {disease: np.random.random() for disease in DISEASE_CLASSES.values()}
        fig_chart = create_probability_chart(test_probs)
        print("✅ Gráfico de probabilidades creado")
        
        return True
    except Exception as e:
        print(f"❌ Error en componentes: {str(e)}")
        traceback.print_exc()
        return False

def check_dependencies():
    """Verifica las dependencias principales."""
    print("\n📦 Verificando dependencias...")
    
    dependencies = [
        'tensorflow', 'streamlit', 'numpy', 'pandas', 
        'plotly', 'PIL', 'cv2', 'sklearn'
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'cv2':
                import cv2
            elif dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NO INSTALADO")
            missing.append(dep)
    
    return len(missing) == 0

def check_file_structure():
    """Verifica la estructura de archivos."""
    print("\n📁 Verificando estructura de archivos...")
    
    required_files = [
        "Modelo Entrenado/best_resnet152.h5",
        "src/config.py",
        "src/predictor.py", 
        "src/utils.py",
        "app.py",
        "requirements.txt"
    ]
    
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            missing.append(file_path)
    
    return len(missing) == 0

def main():
    """Función principal de pruebas."""
    print("🏥 DERMOSAN - SISTEMA DE PRUEBAS")
    print("=" * 50)
    
    tests = [
        ("Estructura de archivos", check_file_structure),
        ("Dependencias", check_dependencies),
        ("Carga del modelo", test_model_loading),
        ("Procesamiento de imágenes", test_image_processing),
        ("Predicción completa", test_prediction),
        ("Componentes de interfaz", test_streamlit_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name.upper()} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        print("\n🎉 ¡TODOS LOS TESTS PASARON! El sistema está listo para usar.")
        print("\nPara ejecutar la aplicación:")
        print("   streamlit run app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} pruebas fallaron. Revisar errores arriba.")
        
        if not results[0][1]:  # Estructura de archivos
            print("\n💡 Sugerencia: Verificar que todos los archivos estén en su lugar")
        if not results[1][1]:  # Dependencias
            print("\n💡 Sugerencia: Instalar dependencias con: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
