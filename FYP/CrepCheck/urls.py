from django.urls import path, include
from django.contrib import admin
from django.conf import settings
from . import views
from django.conf.urls.static import static

app_name = 'CrepCheck'
urlpatterns =[
    # path('', views.index, name='index'),
    path('', views.upload_image,name='upload_image'),
    path('guide.html', views.guide, name='guide'),
    path('home.html', views.home, name='home'),
    path('dirt.html', views.dirt, name='dirt'),
    path('crease.html', views.crease, name='crease'),
    path('overall.html', views.overall, name='overall'),
] 
if True:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)