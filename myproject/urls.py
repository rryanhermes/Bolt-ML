from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.index, name='index'),
    path('signup/', views.signup, name='signup'),  # Added the view for signup
    path('auth/', include('myapp.urls')),  # Include your app's URLs
    path('testing/', views.testing, name='testing'),
    path('login/', views.login, name='login'),
]