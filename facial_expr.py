import numpy as np
import cv2

from threading import Thread
import time


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
    face_model = cv2.CascadeClassifier("cascades/face_cascade.xml")
    mouth_model = cv2.CascadeClassifier("cascades/mouth_cascade.xml")
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
            mouth = mouth_model.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors= 25,
                minSize = (25,25),
                flags=cv2.CASCADE_SCALE_IMAGE)
            #mouth classifier
            for(sx,sy,sw,sh) in mouth:
                if( sy > int(y + h*2/3)):
                    continue
                cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
    
    
        #show the frame
        cv2.imshow("frame",frame)
        #waitKey
        k = cv2.waitKey(1) & 0xFF;
        if k == 32:
            c.stop()
            break

if __name__ == '__main__':
    main()
