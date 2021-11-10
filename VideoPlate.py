#!/usr/bin/python
import cv2
import numpy as np
from PlateDetect import Detect
from LicensePlateRecognition import opencvReadPlate
# from SegmentingPlates import Segmenting



cap = cv2.VideoCapture('yolov5/video/12.avi')
if (cap.isOpened()== False):
  print("Error opening video stream or file")
  exit(0)

counter_plate=0;
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    # frame= cv2.resize(frame, (650,600))
    Pts= Detect(frame)
    # Pts=[]

    # text, imgg= opencvReadPlate(frame.copy())
    # print(text)
    if Pts: 
      # print(len(Pts)) 
      for pt in Pts:
        x,y,w,h, _, _= pt

        plate= frame[y:h, x:w]
        # m= Segmenting(plate)
        # m= cv2.resize(m, (450,250), cv2.INTER_AREA)
        # cv2.imshow('Plate Segmenting', plate)
        text, imgg= opencvReadPlate(plate.copy())
        print("TEXT: ", text)
        # print(text)
        # if len(c)!=0: print(len(c))


        # cv2.imwrite("Plate/xpl"+str(counter_plate)+".jpg", plate)
        counter_plate+=1

        color= (255, 0, 0)
        frame = cv2.rectangle(frame, (x,y), (w,h), color, 2)

    cv2.imshow('Frame',frame)

    
    # cv2.imshow('Plate Segmenting',ss)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break


cap.release()
cv2.destroyAllWindows()


