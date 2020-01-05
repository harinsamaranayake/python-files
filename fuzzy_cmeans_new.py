# https://github.com/ariffyasri/fuzzy-c-means/blob/master/fuzzy-c-means-scikit-fuzzy-image.ipynb
# https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os
import cv2
import numpy as np
from time import time

def skip_frames(cap,frame_rate,seconds_to_skip):
    # skip frames
    print('wait! skipping frames')
    frame_rate = frame_rate
    seconds_to_skip = seconds_to_skip

    for i in range(frame_rate*seconds_to_skip):
        ret, frame = cap.read()
        # print('skipped frame:\t', i)

    return cap

def change_color_fuzzycmeans(clusters,cluster_membership):
    img = []
    for pix in cluster_membership.T:
        # np.argmax(pix) | larger value INDEX out of the two pixel coordinstes
        img.append(clusters[np.argmax(pix)])
    return img

def bwarea(img):
    row = img.shape[0]
    col = img.shape[1]
    total = 0.0
    for r in range(row-1):
        for c in range(col-1):
            sub_total = img[r:r+2, c:c+2].mean()
            if sub_total == 255:
                total += 1
            elif sub_total == (255.0/3.0):
                total += (7.0/8.0)
            elif sub_total == (255.0/4.0):
                total += 0.25
            elif sub_total == 0:
                total += 0
            else:
                r1c1 = img[r,c]
                r1c2 = img[r,c+1]
                r2c1 = img[r+1,c]
                r2c2 = img[r+1,c+1]
                
                if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                    total += 0.75
                else:
                    total += 0.5
    return total

def imfill(im_th):
    
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    return im_out

def im_clear_border(imgBW):

    # Given a black and white image, first find all of its contours
    radius = 2
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

def bw_area_open(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    # https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html

    img_test = np.zeros([200,200],dtype=np.uint8)
    img_test.fill(255)

    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    # Then if the area is below the theshold then remove it.
    # Note : np.arange(3) > array([0, 1, 2])
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
    # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours
    
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(img_test, contours, idx, (0,0,0), -1)

    return img_test

def get_fuzz_img(frame=None):
    img = frame
    img = org_img = cv2.resize(img, (200,200)).astype(np.uint8)

    shape = np.shape(img)

    img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    cluster = 2

    # Cluster centers.  Data for each center along each feature provided for every cluster (of the `c` requested clusters). 
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)
    print('cntr',np.shape(cntr))
    print('u',np.shape(u))
    print(cntr)
    print(u)

    fuzzy_img = change_color_fuzzycmeans(cntr,u)
    # print('f1',np.shape(fuzzy_img))

    fuzzy_img = np.reshape(fuzzy_img,shape).astype(np.uint8)
    # print('f2',np.shape(fuzzy_img))

    ret, seg_img = cv2.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv2.THRESH_BINARY)
    # print('seg_img',np.shape(seg_img))
    
    seg_img_1d = seg_img[:,:,2]
    # print('seg_img_1d',np.shape(seg_img_1d))

    bwfim = seg_img_1d

    # remove small areas
    # bwfim = bw_area_open(bwfim, 100)
    # print('bwfim',np.shape(bwfim))

    # bwfim = im_clear_border(bwfim)
    # print('bwfim',np.shape(bwfim))
    
    # bwfim = imfill(bwfim)
    # print('bwfim',np.shape(bwfim))

    # print('Bwarea : '+str(bwarea(bwfim)))

    # plt.imshow(bwfim3)
    # name = 'segmented'+str(index)+'.png'
    # plt.savefig(name)    

    return org_img,fuzzy_img,seg_img,bwfim

def view_fuzz_video(video_path=None):
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))

    cap = skip_frames(cap=cap,frame_rate=30,seconds_to_skip=10)

    resize_factor=1

    while(1):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(frame.shape[1]/resize_factor), int(frame.shape[0]/resize_factor)))

        org_img,fuzzy_img,seg_img,bwfim = get_fuzz_img(frame)

        cv2.imshow('org_img',org_img)
        cv2.imshow('fuzzy_img',fuzzy_img)
        cv2.imshow('seg_img',seg_img)
        cv2.imshow('bwfim',bwfim)
        cv2.waitKey(0)

        break

if __name__ == "__main__":
    video_name= "DJI_0004.MOV" #"DJI_0112_S_1.MOV" #"DJI_0002_S_1.MOV" #"DJI_0004.MOV" #"DJI_0002_S_1.MOV" #"DJI_0010.MOV"
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/down/"+video_name
    view_fuzz_video(video_path)

    # image_path="/Users/harinsamaranayake/Desktop/test.png"
    # img = cv2.imread(image_path)
    # org_img,fuzzy_img,seg_img,bwfim = get_fuzz_img(frame=img)
    # cv2.imshow('org_img',org_img)
    # cv2.imshow('fuzzy_img',fuzzy_img)
    # cv2.imshow('seg_img',seg_img)
    # cv2.imshow('bwfim',bwfim)
    # cv2.waitKey(0)