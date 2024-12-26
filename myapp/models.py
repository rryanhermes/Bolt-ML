from django.db import models
from django.contrib.auth.models import User

class UserModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=500)
    metrics = models.JSONField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.model_type} ({self.user.username})"
