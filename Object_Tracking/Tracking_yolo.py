import cv2
import numpy as np
from kalmanfilter import KalmanFilter

# cap = cv2.VideoCapture('/home/adeel/Work/opencv/ObjectTracking/yolov3_tracking/Tourist Crossing The Street.mp4')
cap = cv2.VideoCapture(1)
kf = KalmanFilter()

whT = 320
confThreshold =0.5
nmsThreshold= 0.2

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
            px, py = int(predicted[0]-(w/2)), int(predicted[1]-(h/2))
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            cv2.rectangle(img, (px, py), (px+w,py+h), (255, 0 , 0), 2)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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
    trackObj(outputs,frame)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(150)
    if key == 27:
        break