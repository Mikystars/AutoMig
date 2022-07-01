import cv2
import os
from cameraCV2 import get_csi_stream

width = 1280
height = 720
fps = 30

if os.name == 'nt':
    cap = cv2.VideoCapture(0)
if os.name == 'posix':
    cap = get_csi_stream(width, height, fps)

def mostrar_video():
    if cap.isOpened() == False:
        print('Error al abrir la cámara')
        return
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret == True:
            # Escribe el frame en outpy.avi
            out.write(frame)

            # Muestra el frame en pantalla
            cv2.imshow('Frame', frame)

            # Presionar Q para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    mostrar_video()
