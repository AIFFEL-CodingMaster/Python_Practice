from django.urls import path
from .views import predict

app_name = "vision"

urlpatterns = [
    path("", predict, name='predict')
]