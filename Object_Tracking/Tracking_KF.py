import cv2
import numpy as np

cap = cv2.VideoCapture(1)
_, f1 = cap.read()
gray_f1 =cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize=(15,15),
                 maxLevel = 2,
                 criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

def select_pt(event, x,y, flags, params):
    global pt, pt_select, pt1
    if event == cv2.EVENT_LBUTTONDOWN:
        pt =(x,y)
        pt_select= True
        pt1 = np.array([[x,y]], dtype=np.float32)
        
kf = KalmanFilter()
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_pt)

pt_select = False
pt = ()
pt1 = np.array([[]])


while True:
    _, f2 = cap.read()
    gray_f2 =cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    if pt_select is True:
        
        pt2, status, error = cv2.calcOpticalFlowPyrLK(gray_f1, gray_f2, pt1, None, **lk_params)
        gray_f1 = gray_f2.copy()
        pt1 = pt2

        x, y = pt2.ravel()
        centre = np.array(pt2.ravel())
        x = np.round(x).astype("int")
        y = np.round(y).astype("int")
        predicted = kf.predict(x, y)
        px, py = int(predicted[0]), int(predicted[1])

        cv2.circle(f2, (x,y), 20, (0,255,0), 4)
        cv2.circle(f2, (px,py), 20, (255,0,0), 4)
        

    cv2.imshow("frame",f2)

    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()    
