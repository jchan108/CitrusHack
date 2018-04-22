import numpy as np
import cv2
import math
from threading import Thread
import time

def area(x):
    return x[2] * x[3]

#run video feed on a different thread
class cameraFeed():
    #init function
    def __init__(self,source=0):
        self.stream = cv2.VideoCapture(source)
        _,self.frame = self.stream.read()
        self.stopped = False
    #start function
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    #update function
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            _,self.frame = self.stream.read()
    #read function
    def read(self):
        return self.frame
    #stop function
    def stop(self):
        self.stopped = True


def main():
    
    lower = np.array([-7,121,80])
    upper = np.array([55,193,171])
    face_model = cv2.CascadeClassifier("cascades/face_cascade.xml")
    mouth_model = cv2.CascadeClassifier("cascades/mouth_cascade.xml")
    c = cameraFeed().start()
    ref = c.read()
    while True:
        found = False
        frame = c.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #grab our mouth
            mouths = mouth_model.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors= 25,
                minSize = (25,25),
                flags=cv2.CASCADE_SCALE_IMAGE)
            #mouth classifier
            if len(mouths) < 1:
                continue
            #grab the biggest rectangle
            m = max(mouths,key = area)
            if( m[1] > int(y+ h*2/3)):
                continue
            found = True
            cv2.rectangle(roi_color,(m[0],m[1]),(m[0]+m[2],m[1]+m[3]),(0,255,0),2)
            #try and calculate the curvature of the mouth
        
        if found:
            #show the frame
            roi = roi_color[m[1]:m[1]+m[2],m[0]:m[0]+m[2]]
            roi = cv2.resize(roi, (500,500))
            blurr = cv2.blur(roi,(15,15))
            hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            thresh = cv2.inRange(hsv,lower,upper)
            cv2.imshow("ROI",roi)
            cv2.imshow("hsv",hsv)
            cv2.imshow("thresh",thresh)

        cv2.imshow("frame",frame)
        #waitKey
        k = cv2.waitKey(1) & 0xFF;
        if k == 32:
            c.stop()
            break

if __name__ == '__main__':
    main()
