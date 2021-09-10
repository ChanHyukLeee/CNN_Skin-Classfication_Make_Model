import cv2 as cv
import numpy as np
import imageio
from os import listdir
from os.path import join, isfile, splitext

# extract A* 
def convert_A_extract(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    _,a,_ = cv.split(img_lab)
    a_max = np.max(a)
    a_max_inv = 1 / a_max
    Normalize_a = a_max_inv * a
    # Normalize_a = Normalize_a.astype('uint8')
    return Normalize_a

def threshold(img):
    img_thresh = cv.threshold(img, 0.5, 1,cv.THRESH_BINARY_INV)
    return img_thresh

if __name__ == "__main__":
    image = cv.imread('data/please.jpg')
    A_img = convert_A_extract(image)
    IMG_THRESH = threshold(A_img)
    print(IMG_THRESH)
    cv.imshow('after', IMG_THRESH)
    cv.waitKey(0)