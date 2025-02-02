from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserModel

# Register UserModel
@admin.register(UserModel)
class UserModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'model_type', 'created_at')
    list_filter = ('model_type', 'created_at')
    search_fields = ('name', 'user__username')
