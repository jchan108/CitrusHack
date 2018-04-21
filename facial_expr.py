import numpy as np
import cv2

def main():
    face_model = cv2.CascadeClassifier("cascades/face_cascade.xml")
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_model.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        #show the frame
        cv2.imshow("frame",frame)

        #waitKey
        k = cv2.waitKey(1) & 0xFF;
        if k == 32:
            break

if __name__ == '__main__':
    main()
