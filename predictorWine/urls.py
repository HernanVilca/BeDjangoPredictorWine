from django.urls import path

#from predictor.views import predict

from predictorWine.views import predict

urlpatterns = [
    path('predictor/', predict, name='predict')
]