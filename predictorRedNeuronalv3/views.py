import os
import numpy as np
import joblib
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse

# Definir las rutas para el modelo y el escalador
modelo_path = os.path.join(settings.BASE_DIR, 'predictorRedNeuronalv3', 'modelo_vino.h5')
scaler_path = os.path.join(settings.BASE_DIR, 'predictorRedNeuronalv3', 'scaler_vino.pkl')


# Cargar el modelo y el escalador al inicio
modelo = tf.keras.models.load_model(modelo_path)
scaler = joblib.load(scaler_path)

def predict_wine_quality_red_neuronal_v3(request):
    try:
        # Obtener los datos enviados por el frontend (React o formulario)
        data = request.GET

        # Extraer las características y convertirlas a tipo float
        features = [
            float(data.get('fixed_acidity', 0)),
            float(data.get('volatile_acidity', 0)),
            float(data.get('citric_acid', 0)),
            float(data.get('residual_sugar', 0)),
            float(data.get('chlorides', 0)),
            float(data.get('free_sulfur_dioxide', 0)),
            float(data.get('total_sulfur_dioxide', 0)),
            float(data.get('density', 0)),
            float(data.get('pH', 0)),
            float(data.get('sulphates', 0)),
            float(data.get('alcohol', 0)),
        ]

        # Convertir las características en una matriz numpy y escalar
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)

        # Hacer la predicción
        prediction = modelo.predict(features_scaled)
        quality = int(np.argmax(prediction, axis=1)[0])

        # Enviar la predicción como respuesta JSON
        return JsonResponse({'quality': quality})

    except Exception as e:
        # En caso de error, devolver el mensaje de error
        return JsonResponse({'error': str(e)})
