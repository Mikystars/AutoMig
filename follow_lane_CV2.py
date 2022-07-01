import logging
import math
import cv2
import numpy as np
#from cameraCV2 import get_csi_stream
import serial2arduino as ardu
import hid #Para usar el mando USB

import NanoCam as nano

__CUDA = False
_SHOW_IMAGE = False

class SeguidorDeCarril():

    def __init__(self):
        logging.info('Iniciando SeguidorDeCarril - OpenCV')
        self.curr_steering_angle = 90
        self.curr_speed = 0

    def follow_lane(self, frame):
        ''' Entrada al seguidor de carril
            Param:
                frame: El frame sobre el que se va a detectar el carril y el angulo
            Devuelve: 
                final_frame: Frame con los las lineas sueprpuestas
                
        '''
        show_image("Original", frame)

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):
        '''
        Calcula el angulo de giro y superpone las lineas sobre el frame
        Param: 
            frame: El frame sobre el que calcular el angulo
            lane_lines: Las lineas de carril

        Devuelve: frame con linea central de direccion
        '''

        logging.debug('Girando...')
        if len(lane_lines) == 0:
            # Si no hay carril detectado en el frame, no hace nada
            logging.error('No hay lane_lines detectadas, nada que hacer.')
            return frame

        # Calculo el angulo de giro y lo estabiliza
        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))

        # Imagen de direccion
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        #ardu.turn(self.curr_steering_angle)

        show_image("Direccion", curr_heading_image)

        return curr_heading_image

#---------------------#
# Procesado del frame #
#---------------------#

#Detectar carril:
def detect_lane(frame):
    '''
    Detecta las lineas de un carril en un frame
    
    Devuelve:
        lane_lines: lineas del carril
        lane_lines_image: el frame con las lineas de carril superpuestas
    '''
    logging.debug('Detectando carril...')
    
    #Detecta bordes de carril
    edges = detect_edges(frame)
    show_image('edges', edges)

    # Centra la atencion en la mitad inferior del frame
    cropped_edges = region_of_interest(edges)
    show_image('edges cropped', cropped_edges)

    # Obtiene segmentos de las lineas
    line_segments = detect_line_segments(cropped_edges)
    # La imagen con los segmentos superpuestos
    line_segment_image = display_lines(frame, line_segments)
    show_image("line segments", line_segment_image)

    # Detecta las lineas del carril
    lane_lines = average_slope_intercept(frame, line_segments)
    # La imagen con las lineas del carrril superpuestos
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("lane lines", lane_lines_image)

    return lane_lines, lane_lines_image


def detect_edges(frame):
    '''
    Detecta los bordes de un frame

    Devuelve:
        edges: Frame con los bordes de los carriles
    '''
    # Filtro para ROJO
    if __CUDA:
        hsv = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv.download()
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        show_image("hsv", hsv)

    lower_red = np.array([160, 155, 84])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    show_image("white mask", mask)

    # Canny detecta los bordes
    edges = cv2.Canny(mask, 200, 400)

    return edges

def region_of_interest(canny):
    '''
    
    Param:
        canny: Un frame con bordes de la funcion Canny()

    Devuelve:
        masked_image: el frame sin la mitad superior
    '''
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # Poligono que tapa la parte superior de la imagen

    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    show_image("mask", mask)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def detect_line_segments(cropped_edges):
    '''
    Detecta los segmentos de linea en un frame
    Param:
        cropped_edges: El frame de bordes cortado a la mitad 
    Devuelve:
        line_segments: Los segmentos de linea detectados
    '''
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('Detectado line_segment:')
            logging.debug("%s de longitud %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


def average_slope_intercept(frame, line_segments):
    """
    Combina los segmentos de linea en 1 o 2 lineas de carril
    Si todas las pendientes de las lineas son < 0: Solo hay linea de carril Izquierda
    Si todas las pendientes de las lineas son > 0: Solo hay linea de carril Derecha
    Param:
        frame: el frame a tratar
        line_segments: los segmentos de linea que hay en el frame

    Devuelve:
        line_lines: Las lineas calculadas con los segmentos. Puede ser 1 o 2 
    """
    if __CUDA:
        frame = frame.download()
    
    lane_lines = []
    if line_segments is None:
        logging.info('No hay line_segment detectado')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # El segmento de linea de carril izquierdo deberia estar en los 2/3 derechos del frame
    right_region_boundary = width * boundary # El segmento de linea de carril derecho debería de estar en los 2/3 izquierdos del frame

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('Saltando linea vertical (pendiente=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('Lineas de carril: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def compute_steering_angle(frame, lane_lines):
    """ Encuentra el angulo de giro basandose en las coordenadas de las lineas de carril
        
    """
    if len(lane_lines) == 0:
        logging.info('Lineas no detectadas, SKIP')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('1 sola linea de carril detectada. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.00 # 0.0 La camara está en todo el centro. 
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # Encontrar el angulo de giro
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angulo en radianes a la linea vertical central
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angulo engrados a la linea vertical central
    steering_angle = angle_to_mid_deg + 90  # El angulo de giro

    logging.debug('Nuevo angulo de giro: %s' % steering_angle)
    return steering_angle


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    """
    Calcula el nuevo angulo de giro tomando en cuenta el angulo anterior
    para no hacer cambios de angulo muy bruscos. El angulo maximo que cambiará
    será max_angle_deviation de un frame a otro.

    Args:
        curr_steering_angle: El angulo de giro actual
        new_steering_angle: El nuevo angulo obtenido
    Return:
        Angulo estabilizado
    """
    if num_of_lane_lines == 2 :
        # Dos lineas de carril detectadas, desvia mas si es necesario
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # Si solo hay una linea detectada, no desviar mucho
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info('Angulo propuesto: %s, Angulo estabilizado: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle


#---------------------------#
# Funciones auxiliares      #
#---------------------------#
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    if __CUDA:
        frame = frame.download()

    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(255, 0, 0), line_width=5 ):
    '''Muestra la linea de direccion del angulo calculado'''
    
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # (x1,y1) va a ser siempre el centro inferior del frame
    # (x2, y2) hay que calcularlo

    
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    
    # Se dibuja la linea con los puntos calculados
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def length_of_line_segment(line):
    ''' Devuelve la distancia euclidea'''
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # inferior del frame
    y2 = int(y1 * 1 / 2)  # hacer putnos de la mitad del frame abajo

    # colocar la coordenadas dentro del frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


#------------------------------#
#    Funciones principales     #
#------------------------------#
def test_photo(file):
    land_follower = SeguidorDeCarril()
    frame = cv2.imread(file)
    combo_image = land_follower.follow_lane(frame)
    show_image('final', combo_image, True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video(video_file):
    lane_follower = SeguidorDeCarril()
    cap = cv2.VideoCapture(video_file)

    # Saltar primeros frames
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    
    video_w = int(cap.get(3))
    video_h = int(cap.get(4))
    video_overlay = cv2.VideoWriter("video_out/%s_overlay.avi" % (video_file), video_type, 20.0, (video_w, video_h))
    try:
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print('frame %s' % i )
                
                # TODO: Implementar cuda para mejorar los tiempos de procesamiento
                if __CUDA:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    combo_image = lane_follower.follow_lane(gpu_frame)
                
                else:
                    combo_image = lane_follower.follow_lane(frame)
                
                    cv2.imwrite("imgs_out/%s_%03d_%03d.jpg" % (video_file, i, lane_follower.curr_steering_angle), frame)
                    cv2.imwrite("imgs_out/%s_overlay_%03d.png" % (video_file, i), combo_image)
                
                    video_overlay.write(combo_image)
                #show_image("Carril con la linea a seguir", combo_image, True)
            
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
    lane_follower = SeguidorDeCarril()
    # Abro un stream desde la camara CSI
    cap = nano.Camera(flip=0, width=200, height=200, fps=30)
    
    # El tipode video que voy a guardar. Codec: XVID
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
        while 1: #cap.isOpened():
            # Lee el frame actual. ret es si es valido.
            ret = 1 # LA libreria nanocam ya maneja la validez
            frame = cap.read() 

            if ret:
                print('frame %s' % i )

                ps3 = gamepad.read(32)
                speed = ps3[9]
                speed = np.interp(speed, [127, 0], [60, 30])
                ardu.throttle(speed)

                
                if __CUDA:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    combo_image = lane_follower.follow_lane(gpu_frame)
                
                else:
                    combo_image = lane_follower.follow_lane(frame)
                    
                    if record:
                        cv2.imwrite("imgs_out/%s_%03d_%03d.jpg" % ('CSIcamera', i, lane_follower.curr_steering_angle), frame)
                        cv2.imwrite("imgs_out/%s_overlay_%03d.png" % ('CSIcamera', i), combo_image)
                
                        video_overlay.write(combo_image)

                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    break
            else:
                break
    finally:
        ardu.throttle(30)
        cap.release()
        if record:
            video_overlay.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # El mando de PS3, para acelerar
    gamepad = hid.device()
    gamepad.open(0x054c, 0x0268)
    gamepad.set_nonblocking(True)

    #test_video('outpy.avi')
    #test_photo('imagenes/carril.jpg')
    test_camera(0) #Pasar 1 para grabar.
