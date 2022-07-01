import NanoCam as nano
import serial2arduino as ardu

from follow_lane_TF import LaneFollowerTF
from follow_lane_CV2 import SeguidorDeCarril

import hid
import logging
import cv2
import numpy as np

import time

_SHOW_IMAGE = True

def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 1
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def main():

    cam = nano.Camera(flip=0, width=200, height=200, fps=30)

    lane_follower_CV = SeguidorDeCarril()
    lane_follower_TF = LaneFollowerTF()

    while 1:
        if cam.isReady():
        #try:
            frame = cam.read()
            CV2_image = lane_follower_CV.follow_lane(frame)
            TF_image = lane_follower_TF.follow_lane(frame)

            ps3 = gamepad.read(32)
            #Acelerar solo pulsando la X en el mando PS3
            if ps3[3] == 64:
                ardu.throttle(59)

            # PARA USAR EL JOYSTICK DERECHO COMO ACELERADOR
            #speed = ps3[9]
            #speed = np.interp(speed, [127, 0], [60, 30])
            #ardu.throttle(speed)
            
            # El angulo de giro sera la media entre los dos angulos propuestos
            final_steering_angle = (lane_follower_CV.curr_steering_angle + lane_follower_TF.curr_steering_angle) / 2
             
            ardu.turn(int(final_steering_angle))
            
            show_image('Orig', frame)

            show_image('OpenCV', CV2_image)
            show_image('Tensorflow', TF_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    ardu.throttle(30)
    cam.release()
    cv2.destroyAllWindows()

           

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # El mando de PS3, para acelerar
    gamepad = hid.device()
    gamepad.open(0x054c, 0x0268)
    gamepad.set_nonblocking(True)

    main()
