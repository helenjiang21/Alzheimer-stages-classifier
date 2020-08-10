from django.shortcuts import render
from .apps import PredictorConfig
from django.shortcuts import render
from django.conf import settings

from . import forms as F
from .models import Image
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import skimage
import skimage.transform as image_transform

def home(request):    #new
    return render(request, 'home.html')  #new

def about(request):
    return render(request, 'about.html')

def model(request):
    if request.method == "POST":
        context = F.ImgForm(request.POST, request.FILES)

        img = context.save()

        # This will be your classifaction
        result = None
        # --- YOUR MODEL HERE ---

        #path = os.path.join(settings.MODELS, 'saved_model.pb')
        #model = tf.saved_model.load(path)
        model = keras.models.load_model('/Users/jsong/first_project/Django_ML_proj/ADcnn/ADcnn/alexnet_h5.h5')

        # x = np.asarray(img, dtype=np.float64)
        # x = x / 255.0
        #
        #
        # resized_img = image_transform.resize(x, (227, 227))
        image = Image.open(img.image.path).resize((227,227))

        img_arr = np.asarray(image, dtype=np.float64)
        img_arr = img_arr[np.newaxis, :, :, np.newaxis]
        img_arr = img_arr / 255.0
        result = model.predict(img_arr)
        result = np.argmax(result)
        stage = None
        if(result==0):
            stage = "No Detected AD"
        elif(result ==1):
            stage = "Very Mild AD"
        elif(result==2):
            stage = "Mild AD"
        else:
            stage = "Moderate AD"

        return render(request, 'model.html', {'context': context, 'img': img, 'result': result, 'stage': stage})
    else:
        context = F.ImgForm()
    return render(request, 'model.html', {'context': context})