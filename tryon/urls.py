from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('stream/', views.stream_view, name='stream'),
]
