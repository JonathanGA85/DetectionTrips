
# Importando Librerias
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Cargando el modelo preentrenado 
MODEL_PATH = "model/model_Mobilenet_Trips.h5"

# Configurando las dimensiones de las imagenes de entrada    
width_shape = 224
height_shape = 224

# Defiendo las clases para la clasificación de objetos 
names = ['Guisante_Sin_Trips','Guisante_Con_Trips']

# El modelo recibe una imagen y devuelve la prediccion de
# dicha imagen.
class YoloClasificacion:
    def model_prediction(img, model):

        img_resize = resize(img, (width_shape, height_shape))
        x=preprocess_input(img_resize*255)
        x = np.expand_dims(x,axis=0)
    
        preds = model.predict(x)
        return preds
    
