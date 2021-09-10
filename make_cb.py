import cv2 as cv
import math
import numpy as np
import os
from os import listdir
from os.path import join, isfile, splitext
import imageio

def read_img(img_file):
    img = cv.imread(img_file)
    return img

def cb(img):
    Red = []
    Green = []
    Blue = []
    blank = np.zeros(img.shape[:2], dtype="uint8")
    print(blank.shape)
    
    Blue, Red, Green = cv.split(img)

    R_avg = np.mean(Red)
    G_avg = np.mean(Green)
    B_avg = np.mean(Blue)
    
    R_inv = 1 / R_avg
    G_inv = 1 / G_avg
    B_inv = 1 / B_avg
    
    M = max(R_inv, G_inv, B_inv)

    R_scale = (R_inv / M) * Red
    G_scale = (G_inv / M) * Green
    B_scale = (B_inv / M) * Blue

    R_scale = R_scale.astype('uint8')
    G_scale = G_scale.astype('uint8')
    B_scale = B_scale.astype('uint8')
    
    CB = cv.merge([B_scale, R_scale, G_scale])
    return CB

if __name__ == "__main__":
    croppedFaces_Dir = 'cropped_face_dir/'
    originalFile_Dir = 'data/'  
    imageFiles = [join(originalFile_Dir, f) for f in listdir(originalFile_Dir) if isfile(join(originalFile_Dir, f))]
    num_images = len(imageFiles)
    image_counter = 0
    start_index = 0
    end_index = num_images
    for imagefile in imageFiles[start_index:end_index]:
        image_counter += 1
        imageName = splitext(os.path.basename(imagefile))[0]
        image = read_img(imagefile)
        image_cb = cb(image)
        newName = join(croppedFaces_Dir, imageName + "_cb.jpg")
        imageio.imwrite(newName, image_cb)
        if image_counter % 500 == 0: # Report the progress on image processing every 500 images processed
            print("%d images have been processed."%image_counter)