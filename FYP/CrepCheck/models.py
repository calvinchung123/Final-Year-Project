from django.db import models
from datetime import datetime

# Create your models here.

class CrepCheck(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images/')
    body = models.TextField()
    created_at = models.DateTimeField(default=datetime.now, blank=True)
    def __str__(self):
        return self.title

class Shoe(models.Model):
    title = models.CharField(max_length=200)
    pdf = models.FileField(upload_to='shoes/images/')
    def __str__(self):
        return self.title