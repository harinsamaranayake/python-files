#https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

import numpy as np
import cv2 as cv
import cv2 
import skfuzzy as fuzz
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import time

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

def get_fuzz_img(frame=None):
    img = frame
    img = org_img = img #cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2))).astype(np.uint8)

    shape = np.shape(img)

    # img = img.reshape((img.shape[0] * img.shape[1], 3)) # color
    img = img.reshape((img.shape[0] * img.shape[1],1)) # gray
    
    cluster = 2

    # Cluster centers.  Data for each center along each feature provided for every cluster (of the `c` requested clusters). 
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)
    # print('cntr',np.shape(cntr))
    # print('u',np.shape(u))
    # print(cntr)
    # print(u)

    fuzzy_img = change_color_fuzzycmeans(cntr,u)
    # print('f1',np.shape(fuzzy_img))

    fuzzy_img = np.reshape(fuzzy_img,shape).astype(np.uint8)
    # print('f2',np.shape(fuzzy_img))

    ret, seg_img = cv.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv.THRESH_BINARY)
    # print('seg_img',np.shape(seg_img))
    
    # seg_img_1d = seg_img[:,:,0]
    # print('seg_img_1d',np.shape(seg_img_1d))

    # bwfim = seg_img_1d

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

    return org_img,fuzzy_img,seg_img

def get_dendrogram(image=None, resize_factor=5):
    img = image.copy()
    img = cv2.resize(img, (int(img.shape[1] / resize_factor), int(img.shape[0] / resize_factor))) 
    print(img.shape)
    #Note : 3 channels (108, 192, 3) 20736 if doubled RecursionError: maximum recursion depth exceeded while getting the str of an object
    #Note : 1 channels (108, 192, 3) 20736 if doubled RecursionError: maximum recursion depth exceeded while getting the str of an object
    #Note : cv2.imshow displays a RGB image. HSV channel VALUES are considered as RGB VALUES and displayed
    
    # img_disp = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # cv.imshow('test_img', img_disp)
    # cv2.waitKey(0)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0:3]
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    # img = img[:, :, 0:1]
    # img = img.reshape((img.shape[0] * img.shape[1], 1))

    print(img)
    print(img.shape)
    print("Plotting Dendogram")

    start_time = time.time()

    linked = linkage(img, 'average')

    plt.figure(figsize=(15, 7))

    dendrogram(linked,
            orientation='top',
            distance_sort='True',
            show_leaf_counts=False)

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.show()

    print('done')

def view_dense_potical_flow(video_path=None,seconds_to_skip=0,resize_factor=1):
    cap = cv.VideoCapture(cv.samples.findFile(video_path))
    cap = skip_frames(cap=cap,frame_rate=30,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame1 = cv.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))

    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    # set all the values of column 1 of every row to 255
    # hsv[:, 1] = 255
    # hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        frame2 = cv.resize(frame2, (int(frame2.shape[1]/resize_factor), int(frame2.shape[0]/resize_factor)))
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

        # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow
        flow = cv.calcOpticalFlowFarneback(prvs,next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Calculates the magnitude and angle of 2D vectors.
        # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
        # angleInDegrees 0-radians 1-degrees
        mag, ang = cv.cartToPolar(x=flow[...,0], y=flow[...,1], angleInDegrees=1)
        
        # hsv[...,0] = ang*180/np.pi/2
        hsv[...,0] = ang/2 #cv.normalize(ang,None,0,255,cv.NORM_MINMAX) #ang/2 #since 360 exceds 255
        hsv[...,1] = 255
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

        # Obtain dendogram
        # get_dendrogram(image = hsv)
        # break

        # below two outputs the same image. difference is no of channels
        # bgr = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
        # bgr = cv.cvtColor(bgr,cv.COLOR_GRAY2BGR)

        # maximum of the three channels
        max_img = np.max(bgr,axis=2)

        org_img,fuzzy_img,seg_img = get_fuzz_img(max_img)

        # cv.imshow('max',max_img)
        cv.imshow('frame',frame2)
        # cv.imshow('bgr',bgr)
        # cv.imshow('fuzzy_img',fuzzy_img)
        cv.imshow('hsv',hsv[:,:,0])
        cv.imshow('seg_img',seg_img)
        
        # mag=cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        # ang=ang*180/np.pi/2

        # cv.imshow('mag',mag)
        # cv.imshow('ang',ang)
        # cv.imshow('hsv',hsv)

        #single channel display
        # cv.imshow('single_channel_r',bgr[:,:,0])
        # cv.imshow('single_channel_g',bgr[:,:,1])
        # cv.imshow('single_channel_b',bgr[:,:,2])
        
        # gray = cv.cvtColor(hsv,cv.COLOR_BGR2GRAY)
        # cv.imshow('gray',gray)
        
        k = cv.waitKey(1) & 0xff

        if k == 27:
            break
        elif k == ord('q'):
            cv.destroyAllWindows()
            break
        elif k == ord('s'):
            # cv.imwrite('opticalfb.png',frame2)
            # cv.imwrite('opticalhsv.png',bgr)
            pass

        prvs = next
    
    cap.release()

if __name__ == "__main__":
    video_name= "DJI_0004.MOV" #"DJI_0112_S_1.MOV" #"DJI_0002_S_1.MOV" #"DJI_0004.MOV" #"DJI_0002_S_1.MOV" #"DJI_0010.MOV"
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/down/"+video_name
    view_dense_potical_flow(video_path=video_path,seconds_to_skip=0,resize_factor=2)

