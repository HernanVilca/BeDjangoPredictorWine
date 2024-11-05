from django.urls import path

#from predictor.views import predict

#from predictorWine.views import predict

from predictorRedNeuronal.views import predict_wine_quality_red_neuronal
urlpatterns = [
    path('predictorredneuronalxx/', predict_wine_quality_red_neuronal, name='predict')
]