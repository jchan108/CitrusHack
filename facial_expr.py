import numpy as np
import cv2

def main():
    face_model = cv2.CascadeClassifier("cascades/face_cascade.xml")
    smile_model = cv2.CascadeClassifier("cascades/smile_cascade.xml")
    
    cap = cv2.VideoCapture(0)

    while True:
        _,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_model.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            smiles = smile_model.detectMultiScale(
                roi_gray,
                scaleFactor= 1.7,
                minNeighbors= 22,
                minSize = (25,25),
                flags=cv2.CASCADE_SCALE_IMAGE)
            for(sx,sy,sw,sh) in smiles:
                cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
    
        #show the frame
        cv2.imshow("frame",frame)

        #waitKey
        k = cv2.waitKey(1) & 0xFF;
        if k == 32:
            break

if __name__ == '__main__':
    main()
