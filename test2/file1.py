import numpy as np
import cv2

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('TestVideo.avi')

while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',gray)
    cv2.imshow('img1',img)
    k = cv2.waitKey(1)
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()
