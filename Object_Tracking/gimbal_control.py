import cv2
import numpy as np
from kalmanfilter import KalmanFilter
import serial

cap = cv2.VideoCapture(0)
kf = KalmanFilter()
ser = serial.Serial('COM8', baudrate=9600) 

whT = 320
confThreshold =0.5
nmsThreshold= 0.2
PI = 3.14
str = ""

classFile = r'D:\Python\openCV\Object_Tracking-main\Object_Tracking-main\coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = r"D:\Python\openCV\Object_Tracking-main\Object_Tracking-main\yolov3-320.cfg"
modelWeights = r"D:\Python\openCV\Object_Tracking-main\Object_Tracking-main\yolov3-320.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def trackObj(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        if classNames[classIds[i]] == 'person':
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cx = int(x + (w/2))
            cy = int(y + (h/2))
            predicted = kf.predict(cx, cy)
            px, py = int(predicted[0]), int(predicted[1])
            return px, py
            

while True:
    ret, frame = cap.read()
    if ret is False:
        print('Video error')
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT,whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    outputNames = []

    for i in output_layers_indices:
         outputNames.append(layerNames[i-1])
    
    outputs = net.forward(outputNames)
    x, y = trackObj(outputs,frame)

    anglex = int((x - whT)/whT*180/PI)
    angley = int((y - whT)/whT*180/PI)

    str = ""
    str += f"{anglex}" + "," + f"{angley}" + "$" + "\r"
    print(str)
    ser.write(str.encode())

    key = cv2.waitKey(1000)
    if key == 27:
        break

ser.close()