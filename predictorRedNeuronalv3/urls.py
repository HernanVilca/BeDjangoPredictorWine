from django.urls import path

#from predictor.views import predict

#from predictorWine.views import predict

from predictorRedNeuronalv3.views import predict_wine_quality_red_neuronal_v3

urlpatterns = [
    path('predictorredneuronalxxv3/', predict_wine_quality_red_neuronal_v3, name='predictv3')
]