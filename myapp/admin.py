from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile, UserModel

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Premium Status'

class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_premium_user')
    
    def is_premium_user(self, obj):
        return obj.userprofile.is_premium
    is_premium_user.short_description = 'Premium'
    is_premium_user.boolean = True

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

# Register UserModel
@admin.register(UserModel)
class UserModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'model_type', 'created_at')
    list_filter = ('model_type', 'created_at')
    search_fields = ('name', 'user__username')
