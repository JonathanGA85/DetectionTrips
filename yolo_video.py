# Importando Librerias
import numpy as np
import cv2
import time
import os

class YoloVideo:
# Definiendo las variables 
    def __init__(self,videos,weights,cfg,names,vn):
        self.videos = videos
        self.weights = weights
        self.cfg = cfg
        self.names = names
        self.vn = vn
 # Configuracion del modelo Darknet y el algoritmo Yolo
    def run_model(self): 
        video = self.videos #input
        writer = None
        h, w = None, None
        with open(self.names) as f:
            labels = [line.strip() for line in f]
        network = cv2.dnn.readNetFromDarknet('./model/yolov3.cfg','./model/yolov3.weights')
       
        layers_names_all = network.getLayerNames()
        layers_names_output = \
            [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

        probability_minimum = 0.5

        threshold = 0.3
       
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
       
        lps = []
        f = 0
        t = 0
        while True:
            # Captura de fotograma en fotograma
            ret, frame = video.read()

            # Si no capturo el fotograma por
            # ejemplo al final del video, entonces
            # rompemos el bucle
            if not ret:
                break

            # Obtenemos las dimensiones espaciales del fotograma, 
            # una sola vez desde el inicio todos los demas
            # fotogramas tienen la misma dimension
            if w is None or h is None:
                
                h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)


            # Implementando el pase hacia adelante con nuestro blob y solo a través de capas de salida
            # Calculando el tiempo necesario para pase hacia adelante
            network.setInput(blob)  # Configurando blob como entrada a la red
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()
            
            # Incremento de los contadores de fotogramas y tiempo total
            f += 1
            t += end - start

            # Mostrando el tiempo invertido para un solo fotograma actual
            print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
            yield('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

            bounding_boxes = []
            confidences = []
            class_numbers = []

            # Pasando por todas las capas de salida después de la pasada de avance
            for result in output_from_network:
                # Pasando por todas las detecciones de la capa de salida actual
                for detected_objects in result:
                    scores = detected_objects[5:]
                    class_current = np.argmax(scores)
                    confidence_current = scores[class_current]
                    if confidence_current > probability_minimum:
                        box_current = detected_objects[0:4] * np.array([w, h, w, h])

                        # Ahora, desde el formato de datos YOLO, podemos obtener las coordenadas 
                        # de la esquina superior izquierda que son x_min e y_min
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        # Agregar resultados a listas preparadas
                        bounding_boxes.append([x_min, y_min,
                                               int(box_width), int(box_height)])
                        confidences.append(float(confidence_current))
                        class_numbers.append(class_current)
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                       probability_minimum, threshold)
            if len(results) > 0:
                # Pasando por índices de resultados
                for i in results.flatten():
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                    colour_box_current = colours[class_numbers[i]].tolist()
                    # Dibujar cuadro delimitador en el marco actual
                    roi = frame[y_min:y_min+box_height+10,x_min:x_min+box_width+10]  
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  colour_box_current, 2)
                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                           confidences[i])
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                
            if writer is None:
            
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter('result'+self.vn, fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)
            name = 'result'+self.vn
            writer.write(frame)
        # Visualizando el numero total de fotogrmas y tiempo empleado en la detecion
        # de objetos 
        print()
        print('Total number of frames', f)
        print('Total amount of time {:.5f} seconds'.format(t))
        print('FPS:', round((f / t), 1))
        video.release()
        writer.release()
        yield name