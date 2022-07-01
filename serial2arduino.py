import keyboard
import serial
import time
#Para mapear los valores de us a grados
from numpy import interp
import cv2


ser = serial.Serial('/dev/ttyUSB0', 250000, timeout = 0.05)

def throttle(value):
    command = 'V' + str(value) + '\n'
    ser.write(command.encode())

def turn(angle):
    #angle es el mapeo de 1200-1900 a 45-135.
    microseconds = interp(angle, [45, 135], [1200, 1900])
    command = 'G' + str(microseconds) + '\n'
    ser.write(command.encode());
        

if __name__ == '__main__':
    while True:
    
        if keyboard.is_pressed("w"):
            print("Adelante")
            throttle(60)
        else:
             throttle(30)

        if keyboard.is_pressed("o"):
            print("Izquierda")
            turn(45)
        elif keyboard.is_pressed("p"):
            print("Derecha")
            turn(135)
        else:
            turn(90)

        time.sleep(0.05)
