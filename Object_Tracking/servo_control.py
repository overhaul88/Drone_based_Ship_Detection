import cv2
import numpy as np
from kalmanfilter import KalmanFilter
import serial

kf = KalmanFilter()
ser = serial.Serial('COM8', baudrate=9600) 

# flag = 1
i = 1
str = ""
            

while True:

    anglex = int(2*i)
    angley = int(2*i)

    str = ""
    str += f"{anglex}" + "," + f"{angley}" + "$" + "\r"
    print(str)
    ser.write(str.encode())

    i += 1
    # flag *= -1

    key = cv2.waitKey(1000)
    if i > 99:
        break

ser.close()