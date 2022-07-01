import sys
import getopt

import cv2
import numpy as np
import logging
import math
from tensorflow.keras.models import load_model
#from follow_lane_CV2 import SeguidorDeCarril
import NanoCam as nano

import serial2arduino as ardu

_SHOW_IMAGE = False

__CUDA = False

class LaneFollowerTF(object):

    def __init__(self,
                 model_path = 'models/lane_navigation_final.h5'):
        logging.info('Iniciando Seguir de Carril con Tensorflow')

        self.curr_steering_angle = 90
        self.model = load_model(model_path)

    def follow_lane(self, frame):
        show_image("Original", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("Angulo de giro actual = %d" % self.curr_steering_angle)

        ardu.turn(self.curr_steering_angle)


        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame

    def compute_steering_angle(self, frame):
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = self.model.predict(X)[0]

        logging.debug('Nuevo angulo de giro: %s' % steering_angle)
        return int(steering_angle + 0.5)


def img_preprocess(image):
    '''Preprocesado de la imagen con los parámetros de la red neuronal'''
    height, _, _ = image.shape
    image = image[int(height/2):,:,:] 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) 
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66))
    image = image / 255
    return image

def display_heading_line(frame, steering_angle, line_color=(0, 255, 0), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


############################
# Funciones principales
############################
def test_image(file):
    lane_follower_tf = LaneFollowerTF()
    frame = cv2.imread(file)
    combo_image = lane_follower_tf.follow_lane(frame)
    show_image('final', combo_image, True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video(video_file):
    lane_follower_tf = LaneFollowerTF()

    video_file = video_file + '.avi'
    cap = cv2.VideoCapture(video_file)

    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    
    video_w = int(cap.get(3))
    video_h = int(cap.get(4))
    video_overlay = cv2.VideoWriter("video_out\\%s_overlay.avi" % (video_file), video_type, 20.0, (video_w, video_h))
    try:
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print('frame %s' % i )
                
                #TODO: Implementar cuda para mejorar el rendimiento
                if __CUDA:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    combo_image = lane_follower_tf.follow_lane(gpu_frame)
                
                else:
                    combo_image = lane_follower_tf.follow_lane(frame)
                
                    #cv2.imwrite("imgs_out\\%s_%03d_%03d.jpg" % (video_file, i, lane_follower.curr_steering_angle), frame)
                    #cv2.imwrite("imgs_out\\%s_overlay_%03d.png" % (video_file, i), combo_image)
                
                    #video_overlay.write(combo_image)
                show_image("Road with Lane line", combo_image, True)
            
                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    break
            else:
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


def test_camera(record=0):
    # Creo un seguidor de carril
    lane_follower_tf = LaneFollowerTF()
    # Abro un stream desde la cmara CSI a 320x240 y 20fps
    cap = nano.Camera(flip=0, width=200, height=200, fps=30)

    # El tipode video que voy a guardar. Codec: H264
    video_type = cv2.VideoWriter_fourcc(*'XVID')
    
    # El ancho y alto del stream
    video_w = cap.width
    video_h = cap.height
    video_fps = cap.fps
    
    if record:
        # El overlay del video. Mismo ancho x alto que la captura y 20 fps.
        video_overlay = cv2.VideoWriter("video_out/%s_overlay.avi" % ('CSIcamera'), video_type, video_fps, (video_w, video_h))
    try:
        i = 0
        #TODO: Quitar este bucle, nanocam compruba ya que esté abierta la camara
        while 1:
            # Lee el frame actual. ret es si es valido.
            #TODO: Quitar el ret, la librería se encarga ahora de comprobar la validez
            ret = 1 # La nanocam lo comprueba
            frame = cap.read()
            if ret:
                print('frame %s' % i )
                
                
                combo_image = lane_follower_tf.follow_lane(frame) 
                
                if record:
                    video_overlay.write(combo_image)

                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        if record:
            video_overlay.release()
        cv2.destroyAllWindows()

def main(test_type, file_name):
    if test_type == 0:
        test_camera()

    elif test_type == 1:
        test_video(file_name)

    elif test_type == 2:
        test_image(file_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    mode = 2
    file_path = 'lane_test_imgs/Lane02.jpg'

    try:
        options, remainder = getopt.gnu_getopt(
            sys.argv[1:],
            'm:f:h',
            ['mode=',
            'file=',
            'help'
            ])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)

    #print('OPTIONS   :', options)

    for opt, arg in options:
        if opt in ('-m', '--mode'):
            print('Modo: ' + str(arg))
            mode = int(arg)

        if opt in ('-f', '--file'):
            print('Archivo: ' + str(arg))
            file_path = str(arg)

        if opt in ('-h', '--help'):
            print('''
            --- Ayuda para el seguidor de carril con TensorFlow ---
            Argumentos:

            -m [0, 1, 2]:
            (0: Detectar en la camara,
             1: Detectar en archivo de video,
             2: Detectar en archivo de imagen)

             -f [ruta_fichero]:
             (La direccion de fichero de video o imagen a tratar)

             -h:
             (Muestra esta información de ayuda)

             NOTA: Si no se indica modo ni fichero, por defecto se toma
             '-m 2 -f lane_test_imgs/Lane02.jpg' como argumentos.
            ''')
            sys.exit(0)


    main(mode, file_path)


