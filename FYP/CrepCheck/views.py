from django.shortcuts import render
from django.http import HttpResponseRedirect
from datetime import datetime
from .models import CrepCheck
from .forms import UploadImageForm
from .shoe_grader import grade_shoe
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
# Create your views here.



def handle_uploaded_image(f):
    file_path = 'media/' + f.name
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def index(request):

    return render(request, 'CrepCheck/index.html')

def guide(request):

    return render(request, 'crepcheck/guide.html')

def home(request):

    return render(request, 'crepcheck/home.html')

def dirt(request):

    return render(request, 'crepcheck/dirt.html')

def crease(request):

    return render(request, 'crepcheck/crease.html')


def overall(request):

    return render(request, 'crepcheck/overall.html')



def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_path =handle_uploaded_image(request.FILES['image'])
            grade = grade_shoe(image_path)
            ograde=grade[0]
            cgrade=grade[1]
            #image_path ="static/img/elephant.jpg" 
            return render(request, 'CrepCheck/upload.html', {'form': form, "ograde": ograde, "cgrade":cgrade, "image_path":image_path,})
    else:
        form = UploadImageForm()
    return render(request, 'CrepCheck/upload.html', {'form': form})
'''
'''