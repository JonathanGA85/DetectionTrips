# Importando Librerias
import streamlit as st
import os
import cv2
import numpy as np
import searchconsole
import matplotlib.pyplot as plt
from yolo_video import YoloVideo
from yolo_images import detect_objects
import tempfile
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from yolo_clasification import YoloClasificacion
from PIL import Image

# Configuracion del Batch
st.sidebar.subheader('Maestria Telcomunicaciones')

#######Configuracion del Main#########

def main():
    
    selected_box = st.sidebar.selectbox(
    'Elija una de las siguientes opciones:',
    ('Bienvenido','Detección en Imagenes', 'Detección en Video', 'Detection Real Time','Estado Cultivo')
    
    )
# Configuracion de Menu de Opciones    
    if selected_box == 'Bienvenido':
        welcome() 
    if selected_box == 'Detección en Imagenes':
        imagenes()
    if selected_box == 'Detección en Video':
        video()
    if selected_box == 'Detection Real Time':
        detection_obj()
    if selected_box == 'Estado Cultivo':
        estado_cultivo()  
        
    st.sidebar.image("./media/escudo.png")
    st.sidebar.image("./media/utn.png")

# Configuracion de Bienvenida al Sistema
def welcome():
    
    st.title('Detección de Trips en el Guisante o Arveja')
                 
    
    st.image("./media/guisante.png", use_column_width=True)
    
    
    st.subheader('Esta es una aplicación que permite detectar la plaga trips y en el guisante o arveja,'
             + ' a partir de archivos digitales como imagenes y videos. También pronóstica el estado actual del cultivo si este se encuentra sano o infectado' + 
             ' ,mediante la implementación de algoritmos y métodos basados en Deep Learning.')

# Metodo para la Deteccion de Object en Imagenes                          
def imagenes():
    st.header("Detección de Trips en Imagenes")
    # Cargar el archivo de imagen a emplear en la deteccion 
    st.write("La API utiliza YOLO(You Only Look Once)que es un algoritmo entrenado para identificar miles de objetos y DarkNet que es una CNN(Red Neuronal Convolucional) como método de aprendizaje profundo o Deep Learning")
    image_file = st.file_uploader('Seleccionar un archivo de imagen', type=['jpg','png','jpeg'])
    submit = st.button('Deteccion Trips')
    # Inicio de la deteccion de objetos
    if submit:
        st.set_option('deprecation.showfileUploaderEncoding', False)
        
        if image_file is not None:
            our_image = Image.open(image_file)  
            detect_objects(our_image)
            
# Metodo para la Deteccion de Object en Videos          
def video():
    st.header("Detección de Trips en Video")  
    # Cargar el archivo de video a emplear en la deteccion
    st.write("La API utiliza YOLO(You Only Look Once)que es un algoritmo entrenado para identificar miles de objetos y DarkNet que es una CNN(Red Neuronal Convolucional) como método de aprendizaje profundo o Deep Learning")
    cfg_vid = './model/yolov3.cfg'
    image_vid = st.file_uploader('Seleccionar un archivo de video', type=['mp4','mov'])
    names_vid = './model/coco.names'
    weights_vid = './model/yolov3.weights'
    submit_vid = st.button('Deteccion Trips')
    # Inicio de la deteccion de objetos
    if submit_vid:
        confidence_slider = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.05)
        nms_slider = st.sidebar.slider('Non-Max Suppression Threshold', 0.0, 1.0, 0.3 , 0.05)
        print(image_vid.name)
        vn = image_vid.name
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(image_vid.read())
        vf = cv2.VideoCapture(tfile.name)
        x = YoloVideo(vf,weights_vid,cfg_vid,names_vid,vn)
        
        for t in x.run_model():
            st.write(t)
            if '.mp4' in t:
                videofile = t
                print(videofile)
                st.write('Total number of frames', t)
        m = os.path.splitext(videofile)[0]
        os.system('ffmpeg -i '+videofile+' -vcodec libx264 '+m+'ffmpeg.mp4')
        st.video(m+'ffmpeg.mp4')
        
# Metodo para el Pronostico del Estado del Cultivo          
def estado_cultivo():
    st.header("Estado del Cultivo") 
    
    st.write("La API utiliza YOLO(You Only Look Once)que es un algoritmo entrenado para identificar miles de objetos y DarkNet que es una CNN(Red Neuronal Convolucional) como método de aprendizaje profundo o Deep Learning")
    model=''
    
    if model=='':
        model = load_model('model/model_Mobilenet_Trips.h5')
    # Cargar el archivo a emplear en el pronostico del estado del cultivo    
    predictS=""
    img_file_buffer = st.file_uploader('Seleccionar un archivo de imagen', type=["png", "jpg", "jpeg"])
    # Inicio del pronostico del estado del cultivo
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))    
        st.image(image, caption="Imagen", use_column_width=False)
   
    if st.button("Predicción"):
         predictS = model_prediction(image, model)
         st.success('EL DIAGNÓSTICO ES: {}'.format(names[np.argmax(predictS)]))
         
if __name__ == '__main__':
	main()