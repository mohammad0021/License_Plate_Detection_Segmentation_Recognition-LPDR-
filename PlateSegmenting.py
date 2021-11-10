#!/usr/bin/python
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.filters import threshold_local


def refine_cut(cuts, no_recursion=False):
    """" refine created cuts (multiple cuts into one) """
    prev_c = 0
    c_bank = []
    temp_c = []
    for c in cuts:
        temp_c.append(c)
        if (c - prev_c) > 10:
            c_bank.append(temp_c)
            temp_c = []
        prev_c = c
    cuts0 = [0, ]
    if no_recursion:
        for i in c_bank:
            cuts0.append(int(np.max(np.array(i))))
    else:
        for i in c_bank:
            cuts0.append(int(np.mean(np.array(i))))
    if no_recursion:
        return cuts0
    else:
        return refine_cut(cuts0, no_recursion=True)


def cut_image(img0, cut_points):
    """ cut image with the given cut points"""
    img = np.copy(img0)
    imgs = []
    for i in range(len(cut_points)):
        imgs.append(img[:, cut_points[i - 1]: cut_points[i], :])
        if i == len(cut_points) - 1:
            imgs.append(img[:, cut_points[i]:, :])
    return imgs


def Segmenting(img_and_masks):
    img = img_and_masks
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th2 = th2.astype(np.uint8) // 255
    horizontal_hist = np.sum(th2, axis=0)
    seg_thresh = np.max(horizontal_hist) * 0.87
    cut_points2 = np.where(horizontal_hist > seg_thresh)[0]

    cut_points2 = refine_cut(cut_points2)
    seged_imgs = cut_image(img, cut_points2)

    return seged_imgs



