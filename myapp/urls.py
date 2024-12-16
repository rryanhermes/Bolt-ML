from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup, name='signup'),
    path('testing/', views.testing, name='testing'),
    path('login/', views.login, name='login'),
    path('login/', views.login, name='login'),
]