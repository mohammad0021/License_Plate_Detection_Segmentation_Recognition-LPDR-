#!/usr/bin/python
import cv2
import torch
import numpy as np
from skimage import io


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    
    rect = order_points(pts)
    
    tl, tr, br, bl = pts
    
    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))
    
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def simplify_contour(contour, n_corners=4):
    n_iter, max_iter = 0, 1000
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            print('simplify_contour didnt coverege')
            return None

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

def transform(Plate, contours):
    approx = simplify_contour(contours[0], n_corners=4)
    if approx is None:
        return Plate;
#         x0, y0 = x_min, y_min
#         x1, y1 = x_max, y_min
#         x2, y2 = x_min, y_max
#         x3, y3 = x_max, y_max
#     else:
    x0, y0 = approx[0][0][0], approx[0][0][1]
    x1, y1 = approx[1][0][0], approx[1][0][1]
    x2, y2 = approx[2][0][0], approx[2][0][1]
    x3, y3 = approx[3][0][0], approx[3][0][1]

    points = [[x0, y0], [x2, y2], [x1, y1],[x3, y3]]
    points = np.array(points)
    crop_mask_img = four_point_transform(Plate, points)
    crop_mask_img= cv2.resize(crop_mask_img, (320, 64), interpolation=cv2.INTER_AREA)
    return crop_mask_img;

def GetContours(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(gray,(5,5), 0.5)
    #gray= cv2.equalizeHist(gray)

    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((2,2),np.uint8)
    edged = cv2.dilate(edged, kernel, iterations = 3)
    edged = cv2.erode(edged, kernel2, iterations = 5)

    (cont, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:1]

    contours = []
    for contour in cont:
        perimeter = cv2.arcLength(contour, True)
        approximationAccuracy = 0.02 * perimeter
        approximation = cv2.approxPolyDP(contour, approximationAccuracy, True)
        contours.append(contour)

    return contours


def Perspective(plate):
    PlateCont= GetContours(plate)
    crop_mask_img= transform(plate, PlateCont)
    return crop_mask_img;
