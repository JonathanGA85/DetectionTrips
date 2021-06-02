# Importación librerias 
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2 = st.beta_columns(2)
    col1.subheader("Imagen Original")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # Algortimo YOLO y Red DarkNet
    net = cv2.dnn.readNetFromDarknet("./model/yolov3.cfg", "./model/yolov3.weights")
    
    # Integración de las labels 
    classes = []
    with open("./model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))   

    # Carga Imagen
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape

    # Deteccion de Objetos(Convirtiendo en Blob)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False) #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    # Muestra de la Información contenida en la variable'out'
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  
            confidence = scores[class_id] 
            if confidence > 0.5:   
            # Deteccion de Objetos
            # Obteniendo las variables de la imagen: center,width,height  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width es el ancho original de la imagen
                h = int(detection[3] * height) #height es la altura original de la imagen
                
                # Rectangulo de coordenadas
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                # Organizando los objetos en matriz para que podamos extraerlos más tarde
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
    print(indexes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            # Obteniendo el nombre del objeto
            label = str.upper((classes[class_ids[i]]))   
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
            items.append(label)

    st.text("")
    col2.subheader("Image del Objecto Detectado")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)
    
    # Visualizado los objetos detectados
    if len(indexes)>1:
        st.success("Objectos Detectados {} - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Objectos Detectados {} - {}".format(len(indexes),[item for item in set(items)]))

