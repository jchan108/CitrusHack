import numpy as np
import cv2

def main():
    face_model = cv2.CascadeClassifier("cascades/face_cascade.xml")
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame",gray)
        k = cv2.waitKey(1) & 0xFF;
        if k == 32:
            break

if __name__ == '__main__':
    main()
