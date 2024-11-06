
import os
import numpy as np
import joblib
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse
from sklearn.exceptions import NotFittedError

# Definir las rutas para el modelo y el escalador
modelo_path = os.path.join(settings.BASE_DIR, 'predictorRedNeuronalv3', 'modelo_vino.h5')
scaler_path = os.path.join(settings.BASE_DIR, 'predictorRedNeuronalv3', 'scaler_vino.pkl')

# Cargar el modelo y el escalador al inicio
modelo = tf.keras.models.load_model(modelo_path)
scaler = joblib.load(scaler_path)

def predict_wine_quality_red_neuronal_v3(request):
    try:
        # Validar que los datos requeridos estén presentes
        required_params = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        for param in required_params:
            if param not in request.GET:
                return JsonResponse({'error': f'El parámetro {param} es obligatorio.'}, status=400)

        # Extraer las características y convertirlas a tipo float
        features = [
            float(request.GET.get('fixed_acidity')),
            float(request.GET.get('volatile_acidity')),
            float(request.GET.get('citric_acid')),
            float(request.GET.get('residual_sugar')),
            float(request.GET.get('chlorides')),
            float(request.GET.get('free_sulfur_dioxide')),
            float(request.GET.get('total_sulfur_dioxide')),
            float(request.GET.get('density')),
            float(request.GET.get('pH')),
            float(request.GET.get('sulphates')),
            float(request.GET.get('alcohol')),
        ]

        # Convertir las características en una matriz numpy y escalar
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)

        # Hacer la predicción
        prediction = modelo.predict(features_scaled)
        quality = int(np.argmax(prediction, axis=1)[0])

        # Enviar la predicción como respuesta JSON con estado 200 OK
        return JsonResponse({'quality': quality}, status=200)

    except ValueError as ve:
        # Capturar errores de conversión de tipos de datos (por ejemplo, si no se puede convertir un valor a float)
        return JsonResponse({'error': f'Error en los valores de entrada: {str(ve)}'}, status=400)
    except NotFittedError as nfe:
        # Capturar errores si el escalador o el modelo no están entrenados correctamente
        return JsonResponse({'error': 'El modelo o el escalador no están entrenados correctamente.'}, status=500)
    except Exception as e:
        # Capturar cualquier otro error interno del servidor
        return JsonResponse({'error': f'Error interno del servidor: {str(e)}'}, status=500)
