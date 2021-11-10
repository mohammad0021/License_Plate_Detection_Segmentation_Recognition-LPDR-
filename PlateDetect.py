#!/usr/bin/python3
import cv2
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

weights= 'yolov5/weights/licence-plate.pt'
model = torch.hub.load('./yolov5', 'custom', path=weights, source='local')  # local repo
imgsz= 640


def Detect(img0):
    results = model(img0, size=imgsz)  # includes NMS

    rect=[]
    if results.xyxy[0].tolist():
        for pt in results.xyxy[0].tolist():
            x,y,w,h,_,_= list(map(int, pt))
            dic=20
            y= y-dic; h=h+dic; x=x-dic; w=w+dic;
            if y-dic<0: y=0
            if h-dic<0: h=0
            if w-dic<0: w=0
            if x-dic<0: x=0
            pt= [x,y,w,h,_,_]
            rect.append(list(map(int, pt)))
    
    return (rect)