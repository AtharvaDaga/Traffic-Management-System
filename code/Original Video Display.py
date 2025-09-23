#---------------------------------------------------IMPORTING LIBRARIES-------------------------------------------------------------------------------
import cv2
import numpy as np


#---------------------------------------------------------------------------------------------------------------------------------
cap=cv2.VideoCapture("F:/traffic Management system/Vehicle-Detection-Classification-and-Counting-main/Videos/1.avi")


while(True):
    ret,frame1=cap.read()
    cv2.imshow("Video Original",frame1)
    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()