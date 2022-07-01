# Ahora se usará el mando USB con joystick analogico
# Se podrán tomar medidas de angilo mas precisas que
# solo pulsando una tecla

import hid
import cv2
import os
from cameraCV2 import get_csi_stream
#import keyboard #Ya no es necesario, se usa hid
import serial2arduino as ardu
import sys
from numpy import interp
from multiprocessing import Process

width = 340
height = 340
fps = 30

# Para las pruebas en windows 10
if os.name == 'nt':
    cap = cv2.VideoCapture(0)
if os.name == 'posix':
    #TODO: Cambiar esta implementación a NanoCam
    cap = get_csi_stream(width, height, fps)

def record_video(name):
    if cap.isOpened() == False:
        print('Error al abrir la cámara')
        return

    #Hay que usar la misma reolución que el original sino falla
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Coded y paramteros del VideoWriter
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    i = 0
    
    while True:
        ret, frame = cap.read()
        turn_angle = 90
        speed = 30
        if ret == True:
            i += 1
             
            pad_register = gamepad.read(32)

            speed = pad_register[9]
            #mapeo de joystick a velocidad en arduino
            speed = interp(speed, [127, 0], [58, 30])
            
            #Usando los dos joysticks para mas precision en el giro
            turn = pad_register[8] #+ pad_register[3] # En el mando de PS3 no hace falta los 2 joysticks
            turn_angle = interp(turn, [0, 255], [45, 135])

            ardu.turn(turn_angle)
            ardu.throttle(speed)

            cv2.imshow('Frame', frame)
            
                      
            cv2.imwrite('imgs_out/%s_%03d_%03d.jpg' % (name, i, turn_angle), frame)
             # Escribe el frame en outpy.avi
            out.write(frame)

            # Muestra el frame en pantalla
            # cv2.imshow('Frame', frame)
            
            # Pulsar triangulo para acabar
            if pad_register[3] == 16:
                break
            # Presionar Q para salir

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    gamepad = hid.device()
    # Los valores para referenciar al mando USB
    gamepad.open(0x054c, 0x0268)
    # Si no recibe nada, devuelve 0 y no espera a un cambio
    gamepad.set_nonblocking(True)

    record_video(sys.argv[1])

