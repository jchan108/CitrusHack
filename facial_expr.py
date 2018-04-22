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
    
    lower = np.array([100,100,100])
    upper = np.array([255,255,255])
    face_model = cv2.CascadeClassifier("cascades/face_cascade.xml")
    mouth_model = cv2.CascadeClassifier("cascades/Mouth.xml")
    c = cameraFeed().start()
    ref = c.read()
    while True:
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
            [mx,my,mw,mh] = m
            if(my < int(h *2/3)):
                continue
            
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(255,255,255),2)
            #try and calculate the curvature of the mouth
            x = mx
            xright = mx + int(mw * .7)
            y = my
            w = mw
            wleft = mw - int(mw * .7)
            h = mh
            #create sub images
            roi_l = roi_color[y:y+h,x:x+wleft]
            roi_r = roi_color[y:y+h,xright:xright+wleft]
            #blur
            roi_l = cv2.blur(roi_l,(7,7))
            roi_r = cv2.blur(roi_r,(7,7))
            #resize
            roi_l = cv2.resize(roi_l,(300,300))
            roi_r = cv2.resize(roi_r,(300,300))
            #convert to color
            roi_l_hsv = cv2.cvtColor(roi_l,cv2.COLOR_BGR2HSV)
            roi_r_hsv = cv2.cvtColor(roi_r,cv2.COLOR_BGR2HSV)
            thresh1 = cv2.inRange(roi_l_hsv,lower,upper)
            thresh2 = cv2.inRange(roi_r_hsv,lower,upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
            #apply a close morph
            thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("lefthsv",roi_l_hsv)
            cv2.imshow("righthsv",roi_r_hsv)
            cv2.imshow("left",thresh1)
            cv2.imshow("right",thresh2)

        cv2.imshow("frame",frame)
        #waitKey
        k = cv2.waitKey(1) & 0xFF;
        if k == 32:
            c.stop()
            break

if __name__ == '__main__':
    main()
