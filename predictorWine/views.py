
import os
from django.conf import settings
import joblib
from django.http import JsonResponse

# Definir la ruta del modelo y scaler dentro de la carpeta 'predictor'
modelo_path = os.path.join(settings.BASE_DIR, 'predictorWine', 'modelo_svm_vino.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'predictorWine', 'scaler_vino.pkl')

# Cargar el modelo y el scaler
modelo_svm = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

def predict(request):
    try:
        # Obtener los datos enviados por el frontend (React o formulario)
        data = request.GET

        # Convertir los datos a una lista para poder hacer la predicción
        features = [float(data['fixed_acidity']), float(data['volatile_acidity']), float(data['citric_acid']),
                    float(data['residual_sugar']), float(data['chlorides']), float(data['free_sulfur_dioxide']),
                    float(data['total_sulfur_dioxide']), float(data['density']), float(data['pH']),
                    float(data['sulphates']), float(data['alcohol'])]

        # Convertir a formato numpy y escalar
        features_array = [features]
        features_scaled = scaler.transform(features_array)

        # Hacer la predicción
        prediction = modelo_svm.predict(features_scaled)

        # Devolver la predicción en formato JSON
        return JsonResponse({'quality': int(prediction[0])})

    except Exception as e:
        return JsonResponse({'error': str(e)})
    

#http://127.0.0.1:8000/predictorWine/predictor/?fixed_acidity=7.4&volatile_acidity=0.7&citric_acid=0.0&residual_sugar=1.9&chlorides=0.076&free_sulfur_dioxide=11.0&total_sulfur_dioxide=34.0&density=0.9978&pH=3.51&sulphates=0.56&alcohol=9.4
