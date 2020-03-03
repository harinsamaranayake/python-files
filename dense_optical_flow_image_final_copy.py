#https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

import os
import cv2
import numpy as np 
import skfuzzy as fuzz
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def dense_potical_flow_horizontal_demo(video_path=None,seconds_to_skip=0,resize_factor=1,save_path = None):
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_no = 0

    cap = skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame_no += 1
    frame1 = cv2.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    
    frame_array = []
    mag_max_array = []
    mag_min_array = []

    while(1):
        ret, frame2 = cap.read()
        frame_no += 1

        frame2 = cv2.resize(frame2, (int(frame2.shape[1]/resize_factor), int(frame2.shape[0]/resize_factor)))
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        mag, ang = cv2.cartToPolar(x=flow[...,0], y=flow[...,1], angleInDegrees=1)
        
        mag_max_array.append(np.amax(mag))
        mag_min_array.append(np.amin(min))
        print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))
        
        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        # maximum of the three channels in an rgb image
        max_img = np.max(bgr,axis=2)

        # fuzzy cmeans
        org_img,fuzzy_img,seg_img = get_fuzz_img(max_img)

        # obtaining the angle values only in predicted water regions
        filtered_angle = hsv[...,0] & seg_img

        # Note : by using the AND operator with segmented image, non water regions get a angle = 0
        # But some original water reigions may have angle 0 or near to 0 (0 not observed but 0e7 0e8 observed)
        # Hypothesis : angle is always > 0
        # So the values between 0 and 1  are assinged with 1
        filtered_angle2 = filtered_angle.copy()
        filtered_angle2[( 0 < filtered_angle2) & (filtered_angle2 < 1)] = 1
        
        freq_a, bins_a = np.histogram(filtered_angle2, bins = [0,1,32,64,96,128,160,192,224,256])
        freq_m, bins_m = np.histogram(mag, bins = [0,1,2,3,4,5,6,7,8])

        # based on frame size
        bin_threshold = 2000

        if(np.amin(freq_m[1:])>= bin_threshold ):
            print('8 directions detected with min threshold')

        # water - black | non-water - white
        # mag((225<=ang<=315) & (mag>=gradiant_tresh)) = 255
        # mag((225>=ang>=315) & (mag>=gradiant_tresh)) = 0

        cv2.imshow('original frame',frame2)
        cv2.imshow('next',next)
        cv2.imshow('hsv',hsv)
        cv2.imshow('bgr',bgr)
        cv2.imshow('max',max_img)
        cv2.imshow('filtered_angle',filtered_angle)
        cv2.imshow('fuzzy_img',fuzzy_img)
        cv2.imshow('seg_img',seg_img)

        frame_array.append(bgr)

        prvs = next

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()

    mag_max =  np.amax(mag_max_array)
    mag_min =  np.amin(mag_min_array)      

    return frame_array,fps,mag_max,mag_min

def get_mag_max_min_video(video_path = None):
    video_path = video_path

    mag_max_array = []
    mag_min_array = []

    video_list = next(os.walk(video_path))[2]

    if '.DS_Store' in video_list:
        video_list.remove('.DS_Store')

    for video in video_list:
        video_name = video
        print(video_name)

        video = video_path + video_name
        result_video_save_path = video_path + "result_" + video_name + video_format
        result_image_save_path = video_path

        frame_array,fps,mag_max,mag_min = view_dense_potical_flow_static(video_path=video,seconds_to_skip=0,resize_factor=2,save_path = result_image_save_path)
        mag_max_array.append(mag_max)
        mag_min_array.append(mag_min)
        print(mag_max,mag_min)
        # save_video(frame_array,fps = fps ,save_path = result_video_save_path)

    mag_max = 0
    mag_min = 0
    mag_max =  max(mag_max_array)
    mag_min =  min(mag_min_array)  
    print('mag_max ',mag_max,' mag_max ',mag_min)

def skip_frames(cap,seconds_to_skip):
    # skip frames in a video
    print('wait! skipping frames')

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    seconds_to_skip = seconds_to_skip

    for i in range(fps*seconds_to_skip):
        ret, frame = cap.read()

    return cap

def save_video(frame_array=None, fps=0, save_path=None):
    # https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html

    out = cv2.VideoWriter(filename=save_path, fourcc=cv2.VideoWriter_fourcc(
        *'DIVX'), fps=fps, frameSize=(960, 540), isColor=1)

    frame_array = frame_array

    for i in range(len(frame_array)):
        out.write(frame_array[i])
        print('wrote_to_video:\t', i)

    out.release()

    print("\nVideo Saved")

def display_text_on_frame(frame = None):
    frame = frame

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame,'Water Detected!', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

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

    ret, seg_img = cv2.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv2.THRESH_BINARY)
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

def view_dense_potical_flow_S1(video_path=None,seconds_to_skip=0,resize_factor=1):
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

        # ang = cv.normalize(ang,None,0,255,cv.NORM_MINMAX)
        print('mag',np.amax(mag),'ang',np.amax(ang))
        
        # hsv[...,0] = ang*180/np.pi/2 #radians | ang/2 degrees | both gives a max of 180 | ang/2 #since 360 exceds 255
        # hsv[...,0] = cv.normalize(ang,None,0,255,cv.NORM_MINMAX) # issue here is even if the range is 0 to 10 it is scaled to 0 to 255
        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

        # Obtain dendogram
        # get_dendrogram(image = hsv)
        # break

        # below two outputs the same image. difference is no of channels
        # bgr = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
        # bgr = cv.cvtColor(bgr,cv.COLOR_GRAY2BGR)

        # maximum of the three channels in an rgb image
        max_img = np.max(bgr,axis=2)

        org_img,fuzzy_img,seg_img = get_fuzz_img(max_img)

        # cv.imshow('max',max_img)
        cv.imshow('frame',frame2)
        cv.imshow('bgr',bgr)
        # cv.imshow('fuzzy_img',fuzzy_img)
        # cv.imshow('hsv',hsv[:,:,2])
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
        
        # break
    
    cap.release()

def view_dense_potical_flow_S2(video_path=None,seconds_to_skip=0,resize_factor=1,save_path = None):
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_no = 0

    cap = skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame_no += 1
    frame1 = cv2.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    
    frame_array = []
    mag_max_array = []
    mag_min_array = []

    while(1):
        ret, frame2 = cap.read()
        frame_no += 1

        if(not ((frame_no%59==0) or (frame_no%60==0) or (frame_no%61==0))):
            continue
        
        print (frame_no)

        frame2 = cv2.resize(frame2, (int(frame2.shape[1]/resize_factor), int(frame2.shape[0]/resize_factor)))
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Calculates the magnitude and angle of 2D vectors.
        # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
        # angleInDegrees 0-radians 1-degrees
        mag, ang = cv2.cartToPolar(x=flow[...,0], y=flow[...,1], angleInDegrees=1)
        
        mag_max_array.append(np.amax(mag))
        mag_min_array.append(np.amin(min))
        print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))

        # next_hsv = cv.cvtColor(frame2,cv.COLOR_BGR2HSV)
        # print('hsv',next_hsv.shape,next_hsv[:,:,0].shape,np.amax(next_hsv[:,:,0]),np.amin(next_hsv[:,:,0]))
        
        # mag,ang rescale
        # ang[( 0 < ang) & (ang <= 1)] = 1
        # mag[( 0 < mag) & (mag < 0.5)] = 0
        # print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))
        
        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        # maximum of the three channels in an rgb image
        max_img = np.max(bgr,axis=2)

        # fuzzy cmeans
        org_img,fuzzy_img,seg_img = get_fuzz_img(max_img)

        # obtaining the angle values only in predicted water regions
        filtered_angle = hsv[...,0] & seg_img

        # Note : by using the AND operator with segmented image, non water regions get a angle = 0
        # But some original water reigions may have angle 0 or near to 0 (0 not observed but 0e7 0e8 observed)
        # Hypothesis : angle is always > 0
        # So the values between 0 and 1  are assinged with 1
        filtered_angle2 = filtered_angle.copy()
        filtered_angle2[( 0 < filtered_angle2) & (filtered_angle2 < 1)] = 1
        
        # freq, bins = np.histogram(ang, bins = [0,45,90,135,180,225,270,315,360])
        # freq2, bins2 = np.histogram(hsv[...,0], bins = [0,1,32,64,96,128,160,192,224,256])
        # freq3, bins3 = np.histogram(hsv[...,2], bins = [0,64,128,192,256])
        # freq4, bins4 = np.histogram(filtered_angle, bins = [0,1,32,64,96,128,160,192,224,256])
        freq5, bins5 = np.histogram(filtered_angle2, bins = [0,1,32,64,96,128,160,192,224,256])
        freq6, bins6 = np.histogram(mag, bins = [0,1,2,3,4,5,6,7,8])

        # print(freq,bins)
        # print(freq2,bins2)
        # print(freq3,bins3)
        # print(freq4,bins4)
        print('\n',bins5,freq5)
        print('\n',bins6,freq6)

        bin_threshold = 2000
        if(np.amin(freq5[1:])>= bin_threshold ):
            print('8 directions detected with min threshold')
            # draw_label(frame2, 'Water Detected!', (20,20), (255,255,255))
            # display_text_on_frame(frame2)

        # ret,hsv_seg = cv.threshold(hsv[:,:,2],64,255,cv.THRESH_BINARY)

        # freq, bins = np.histogram(hsv[...,2], 8)
        # d=plt.hist(hsv[:,:,0], bins=8, facecolor='blue', alpha=0.5)
        # plt.show()

        # Obtain dendogram | Hirachical clustering
        # get_dendrogram(image = hsv)
        # break

        # below two output visualization are the same image. Difference is no of channels BGR2GRAY[1] GRAY2BGR[3]
        # bgr = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
        # bgr = cv.cvtColor(bgr,cv.COLOR_GRAY2BGR)

        # cv2.imshow('original frame',frame2)
        # cv2.imshow('next',next)
        # cv2.imshow('hsv',hsv)
        # cv2.imshow('bgr',bgr)
        # cv2.imshow('max',max_img)
        # cv2.imshow('filtered_angle',filtered_angle)
        # cv2.imshow('fuzzy_img',fuzzy_img)
        # cv2.imshow('seg_img',seg_img)

        if (frame_no%120 == 0):
            save_path_org = save_path + "/ORG/IMG_" + video_name + "_" + str(frame_no) + ".png"
            save_path_pred = save_path + "/PRED/IMG_" + video_name + "_" + str(frame_no) + ".png"
            print(save_path_org)
            print(save_path_pred)
            cv2.imwrite(save_path_org,frame2)
            cv2.imwrite(save_path_pred,bgr)

        frame_array.append(bgr)

        prvs = next

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        
        print()

    cap.release()

    mag_max =  np.amax(mag_max_array)
    mag_min =  np.amin(mag_min_array)      

    return frame_array,fps,mag_max,mag_min

def view_dense_potical_flow_M1(video_path=None,seconds_to_skip=0,resize_factor=1):
    cap = cv.VideoCapture(cv.samples.findFile(video_path))
    
    cap = skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame1 = cv.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))

    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

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
        print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))

        next_hsv = cv.cvtColor(frame2,cv.COLOR_BGR2HSV)
        print('hsv',next_hsv.shape,next_hsv[:,:,0].shape,np.amax(next_hsv[:,:,0]),np.amin(next_hsv[:,:,0]))
        
        # mag,ang rescale
        # ang[( 0 <= ang) & (ang <= 1)] = 1
        # mag[( 0 <= mag) & (mag < 0.5)] = 0
        # print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))
        
        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        
        # maximum of the three channels
        max_img = np.max(bgr,axis=2)

        # fuzzy cmeans
        org_img,fuzzy_img,seg_img = get_fuzz_img(max_img)

        # filtering the angle values in assumed water regions
        # Note : non water regions are indicated with angle = 0. but some water reigions may have angle  = 0
        # Hypothesis : angle is always > 0
        filtered_angle = hsv[...,0] & seg_img

        filtered_angle2 = filtered_angle.copy()
        filtered_angle2[( 0 < filtered_angle2) & (filtered_angle2 < 1)] = 1
        
        freq, bins = np.histogram(ang, bins = [0,45,90,135,180,225,270,315,360])
        # freq2, bins2 = np.histogram(hsv[...,0], bins = [0,1,32,64,96,128,160,192,224,256])
        # freq3, bins3 = np.histogram(hsv[...,2], bins = [0,64,128,192,256])
        freq4, bins4 = np.histogram(filtered_angle, bins = [0,1,32,64,96,128,160,192,224,256])
        freq5, bins5 = np.histogram(filtered_angle2, bins = [0,1,32,64,96,128,160,192,224,256])

        print(freq,bins)
        # print(freq2,bins2)
        # print(freq3,bins3)
        print(freq4,bins4)
        print(freq5,bins5)

        bin_threshold = 2000
        if(np.amin(freq5[1:])>= bin_threshold ):
            print('8 directions detected with min threshold')

        # ret,hsv_seg = cv.threshold(hsv[:,:,2],64,255,cv.THRESH_BINARY)

        # freq, bins = np.histogram(hsv[...,2], 8)
        # d=plt.hist(hsv[:,:,0], bins=8, facecolor='blue', alpha=0.5)
        # plt.show()

        # Obtain dendogram
        # get_dendrogram(image = hsv)
        # break

        # below two outputs the same image. difference is no of channels
        # bgr = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
        # bgr = cv.cvtColor(bgr,cv.COLOR_GRAY2BGR)

        cv.imshow('frame',frame2)
        # cv.imshow('next',next)
        # cv.imshow('hsv',hsv)
        # cv.imshow('bgr',bgr)
        cv.imshow('max',max_img)
        # cv.imshow('filtered_angle',filtered_angle)
        cv.imshow('fuzzy_img',fuzzy_img)
        # cv.imshow('seg_img',seg_img)
        
        k = cv.waitKey(1) & 0xff

        if k == ord('q'):
            cv.destroyAllWindows()
            break

        prvs = next
        
        print()
        # break
    
    cap.release()

def view_dense_potical_flow_s3(video_path=None,seconds_to_skip=0,resize_factor=1,save_path = None):
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_no = 0

    # cap = skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame_no += 1
    frame1 = cv2.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    
    frame_array = []
    mag_max_array = []
    mag_min_array = []

    while(1):
        ret, frame2 = cap.read()

        if (ret == False):
            break

        frame_no += 1

        # if(not ((frame_no%59==0) or (frame_no%60==0) or (frame_no%61==0))):
        #     continue
        
        # print (frame_no)

        frame2 = cv2.resize(frame2, (int(frame2.shape[1]/resize_factor), int(frame2.shape[0]/resize_factor)))
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Calculates the magnitude and angle of 2D vectors.
        # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
        # angleInDegrees 0-radians 1-degrees
        mag, ang = cv2.cartToPolar(x=flow[...,0], y=flow[...,1], angleInDegrees=1)
        
        mag_max = np.amax(mag)
        mag_min = np.amin(mag)

        mag_max_array.append(mag_max)
        mag_min_array.append(mag_min)

        # print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))

        # next_hsv = cv.cvtColor(frame2,cv.COLOR_BGR2HSV)
        # print('hsv',next_hsv.shape,next_hsv[:,:,0].shape,np.amax(next_hsv[:,:,0]),np.amin(next_hsv[:,:,0]))
        
        # mag,ang rescale
        # ang[( 0 < ang) & (ang <= 1)] = 1
        # mag[( 0 < mag) & (mag < 0.5)] = 0
        # print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))
        
        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        # maximum of the three channels in an rgb image
        max_img = np.max(bgr,axis=2)

        # fuzzy cmeans
        org_img,fuzzy_img,seg_img = get_fuzz_img(max_img)

        # obtaining the angle values only in predicted water regions
        # filtered_angle = hsv[...,0] & seg_img

        # Note : by using the AND operator with segmented image, non water regions get a angle = 0
        # But some original water reigions may have angle 0 or near to 0 (0 not observed but 0e7 0e8 observed)
        # Hypothesis : angle is always > 0
        # So the values between 0 and 1  are assinged with 1
        # filtered_angle2 = filtered_angle.copy()
        # filtered_angle2[( 0 < filtered_angle2) & (filtered_angle2 < 1)] = 1
        
        # freq, bins = np.histogram(ang, bins = [0,45,90,135,180,225,270,315,360])
        # freq2, bins2 = np.histogram(hsv[...,0], bins = [0,1,32,64,96,128,160,192,224,256])
        # freq3, bins3 = np.histogram(hsv[...,2], bins = [0,64,128,192,256])
        # freq4, bins4 = np.histogram(filtered_angle, bins = [0,1,32,64,96,128,160,192,224,256])
        # freq5, bins5 = np.histogram(filtered_angle2, bins = [0,1,32,64,96,128,160,192,224,256])
        # freq6, bins6 = np.histogram(mag, bins = [0,1,2,3,4,5,6,7,8])

        # print(freq,bins)
        # print(freq2,bins2)
        # print(freq3,bins3)
        # print(freq4,bins4)
        # print('\n',bins5,freq5)
        # print('\n',bins6,freq6)

        # bin_threshold = 2000
        # if(np.amin(freq5[1:])>= bin_threshold ):
        #     print('8 directions detected with min threshold')
            # draw_label(frame2, 'Water Detected!', (20,20), (255,255,255))
            # display_text_on_frame(frame2)

        # ret,hsv_seg = cv.threshold(hsv[:,:,2],64,255,cv.THRESH_BINARY)

        # freq, bins = np.histogram(hsv[...,2], 8)
        # d=plt.hist(hsv[:,:,0], bins=8, facecolor='blue', alpha=0.5)
        # plt.show()

        # Obtain dendogram | Hirachical clustering
        # get_dendrogram(image = hsv)
        # break

        # below two output visualization are the same image. Difference is no of channels BGR2GRAY[1] GRAY2BGR[3]
        # bgr = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
        # bgr = cv.cvtColor(bgr,cv.COLOR_GRAY2BGR)

        # cv2.imshow('original frame',frame2)
        # cv2.imshow('next',next)
        # cv2.imshow('hsv',hsv)
        # cv2.imshow('bgr',bgr)
        # cv2.imshow('max',max_img)
        # cv2.imshow('filtered_angle',filtered_angle)
        # cv2.imshow('fuzzy_img',fuzzy_img)
        # cv2.imshow('seg_img',seg_img)

        if (frame_no%10 == 0):
            save_path_org = save_path + "/ORG/IMG_" + video_name + "_" + str(frame_no) + ".png"
            save_path_pred = save_path + "/PRED/IMG_" + video_name + "_" + str(frame_no) + ".png"
            print(save_path_org)
            print(save_path_pred)
            cv2.imwrite('/Users/harinsamaranayake/Documents/org.png',frame2)
            cv2.imwrite('/Users/harinsamaranayake/Documents/seg.png',seg_img)
            break

        frame_array.append(bgr)

        prvs = next

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        
        # print()

    cap.release()

    # mag_max_array = mag_max_array.flatten()
    # mag_min_array = mag_min_array.flatten()

    mag_max = 0
    mag_min = 0
    mag_max =  max(mag_max_array)
    mag_min =  min(mag_min_array)      

    return frame_array,fps,mag_max,mag_min

def view_dense_potical_flow_static(video_path=None,seconds_to_skip=0,resize_factor=1,save_path = None, mag_thresh=None):
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_no = 0

    cap = skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame_no += 1
    frame1 = cv2.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    hsv_water_only = np.zeros_like(frame1)
    
    frame_array = []
    mag_max_array = []
    mag_min_array = []

    while(1):
        ret, frame2 = cap.read()

        if (ret == False):
            break

        frame_no += 1

        # .....Skip specific frams.....
        # if(not ((frame_no%59==0) or (frame_no%60==0) or (frame_no%61==0))):
        #     continue

        frame2 = cv2.resize(frame2, (int(frame2.shape[1]/resize_factor), int(frame2.shape[0]/resize_factor)))
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Calculates the magnitude and angle of 2D vectors.
        # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
        # angleInDegrees 0-radians 1-degrees
        mag, ang = cv2.cartToPolar(x=flow[...,0], y=flow[...,1], angleInDegrees=1)
        print('\nmag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang),'\tframe',frame2.shape)
        
        mag_water_only = mag.copy()
        ang_water_only = ang.copy()

        #.....MAgnitude maximum and minimum of current frame......
        mag_max = np.amax(mag)
        mag_min = np.amin(mag)
        mag_max_array.append(mag_max)
        mag_min_array.append(mag_min)

        #.....Directions.....
        ang[(0 <= ang) & (ang < 22.5)] = 1
        ang[(22.5 <= ang) & (ang < 67.5)] = 30
        ang[(67.5 <= ang) & (ang < 112.5)] = 60
        ang[(112.5 <= ang) & (ang < 157.5)] = 90
        ang[(157.5<= ang) & (ang < 202.5)] = 120
        ang[(202.5 <= ang) & (ang < 247.5)] = 150
        ang[(247.5 <= ang) & (ang < 292.5)] = 180
        ang[(292.5 <= ang) & (ang < 337.5)] = 210
        ang[(337.5 <= ang) & (ang <= 360)] = 240

        # When the drone moves up wave speed decreases
        mag_thresh = mag_thresh
        mag[(mag_thresh <= mag)] = 1
        mag[(0 <= mag) & (mag < mag_thresh)] = 0

        print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))
        
        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        #.....Water Only.....
        ang_water_only[(0 <= ang_water_only) & (ang_water_only <= 360)] = 130

        mag_thresh = mag_thresh
        mag_water_only[(mag_thresh <= mag_water_only)] = 1
        mag_water_only[(0 <= mag_water_only) & (mag_water_only < mag_thresh)] = 0

        hsv_water_only[...,0] = ang_water_only/360*255
        hsv_water_only[...,1] = 255
        hsv_water_only[...,2] = cv2.normalize(mag_water_only,None,0,255,cv2.NORM_MINMAX)

        bgr_water_only = cv2.cvtColor(hsv_water_only,cv2.COLOR_HSV2BGR)

        #.....Save Video.....
        frame_array.append(bgr)
        # frame_array.append(bgr_water_only)

        #.....Binning......
        # freq_ang, bins_ang = np.histogram(ang, bins = [0,1,32,64,96,128,160,192,224,256])
        # freq_mag, bins_mag = np.histogram(mag, bins = [0,1,2,3,4,5,6,7,8])
        # print('\n',freq_ang,bins_ang)
        # print('\n',freq_mag,bins_mag)

        # bin_threshold = 2000
        # if(np.amin(freq5[1:])>= bin_threshold ):
        #     print('8 directions detected with min threshold')
        #     draw_label(frame2, 'Water Detected!', (20,20), (255,255,255))
        #     display_text_on_frame(frame2)

        #.....Show......
        # cv2.imshow('original frame',frame2)
        # cv2.imshow('8-directions',bgr)
        # cv2.imshow('water_only',bgr_water_only)

        #.....Save......
        if (frame_no%10 == 0):
            save_path_org = save_path + "/ORG/IMG_" + video_name + "_" + str(frame_no) + ".png"
            save_path_pred = save_path + "/PRED/IMG_" + video_name + "_" + str(frame_no) + ".png"
            print(save_path_org)
            print(save_path_pred)
            cv2.imwrite('/Users/harinsamaranayake/Documents/org.png',frame2)
            cv2.imwrite('/Users/harinsamaranayake/Documents/seg.png',bgr)
            cv2.imwrite('/Users/harinsamaranayake/Documents/water.png',bgr_water_only)
            break

        prvs = next

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()

    #.....Magnitude maximum and minimum of all frames......
    mag_max = 0
    mag_min = 0
    mag_max =  max(mag_max_array)
    mag_min =  min(mag_min_array)      

    return frame_array,fps,mag_max,mag_min

def view_dense_potical_flow_horizontal(video_path=None,seconds_to_skip=0,resize_factor=1,save_path = None):
    cap = cv2.VideoCapture(cv2.samples.findFile(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_no = 0

    cap = skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    ret, frame1 = cap.read()
    frame_no += 1
    frame1 = cv2.resize(frame1, (int(frame1.shape[1]/resize_factor), int(frame1.shape[0]/resize_factor)))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    
    frame_array = []
    mag_max_array = []
    mag_min_array = []

    while(1):
        ret, frame2 = cap.read()

        if(not ret):
            break

        frame_no += 1

        # if(not ((frame_no%59==0) or (frame_no%60==0) or (frame_no%61==0))):
        #     continue

        frame2 = cv2.resize(frame2, (int(frame2.shape[1]/resize_factor), int(frame2.shape[0]/resize_factor)))
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow
        flow = cv2.calcOpticalFlowFarneback(prvs,next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Calculates the magnitude and angle of 2D vectors.
        # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
        # angleInDegrees 0-radians 1-degrees
        mag, ang = cv2.cartToPolar(x=flow[...,0], y=flow[...,1], angleInDegrees=1)
        
        mag_max_array.append(np.amax(mag))
        mag_min_array.append(np.amin(mag))
        print('\nmag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))
        
        # mag,ang rescale | ground pixesl are interested
        ang[(225 <= ang) & (ang <= 315)] = 1
        ang[(225 > ang) & (ang > 1)] = 0
        ang[(315 < ang) & (ang < 360)] = 0

        mag_thresh = 0.5
        mag[(mag_thresh <= mag)] = 1
        mag[(0 <= mag) & (mag < mag_thresh)] = 0

        hsv[...,0] = ang/360*255
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        print('mag',mag.shape,np.amin(mag),np.amax(mag),'\tang',ang.shape,np.amin(ang),np.amax(ang))

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # .....Color Conversion.....
        # If red channel has a value that pixel is made black
        bgr[...,0] = 234
        bgr[...,1] = 207
        bgr[...,0][bgr[...,2] > 0] = 0
        bgr[...,1][bgr[...,2] > 0] = 0
        bgr[...,2][bgr[...,2] > 0] = 0

        # .....Show Image.....
        # cv2.imshow('original frame',frame2)
        # cv2.imshow('bgr',bgr)

        # .....Save Image.....
        if (frame_no%120 == 0):
            save_path_org = save_path + "/ORG/IMG_" + video_name + "_" + str(frame_no) + ".png"
            save_path_pred = save_path + "/PRED/IMG_" + video_name + "_" + str(frame_no) + ".png"
            print(save_path_org)
            print(save_path_pred)
            # cv2.imwrite(save_path_org,frame2)
            # cv2.imwrite(save_path_pred,seg_img)

        # .....Save Video.....
        frame_array.append(bgr)

        prvs = next

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()

    mag_max =  np.amax(mag_max_array)
    mag_min =  np.amin(mag_min_array)      

    return frame_array,fps,mag_max,mag_min

if __name__ == "__main__":

    #..........Hodizontal..........

    # #.....Horizontal Moving 2_5 .....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_horizontal_moving/2_5_m/"
    # video_name = "DJI_0183"
    # video_format = ".MP4"

    # #.....Horizontal Moving 5_10 .....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_horizontal_moving/5_10m/"
    # video_name = "DJI_0174"
    # video_format = ".MP4"

    # #.....Horizontal Moving 10_20 .....
    # # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_horizontal_moving/10_20_m/"
    # # video_name = "DJI_0176"
    # # video_format = ".MP4"

    # #.....Horizontal Moving 20_50 .....
    # # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_horizontal_moving/20_50_m/"
    # # video_name = "DJI_0179"
    # # video_format = ".MP4"

    # # #.....Phantom.....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/phantom/down/non_rain/"
    # video_name = "DJI_0005"
    # video_format = ".MOV"
    # mag_thresh = 0.5


    # video_format_save_as = ".MP4"
    # video = video_path + video_name + video_format
    # result_video_save_as = video_path + "result_w_" + video_name + video_format_save_as
    # result_image_save_as = video_path

    # frame_array, fps, mag_max, mag_min = view_dense_potical_flow_horizontal(video_path=video,seconds_to_skip=0,resize_factor=2,save_path = result_image_save_as)

    # save_video(frame_array,fps = fps ,save_path = result_video_save_as)

    #..........Static..........

    #.....mavic_mini_stable/stable_water/1_2_m.....
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/1_2_m/"
    video_name = "DJI_1580645304000"
    video_format = ".MP4"
    mag_thresh = 1.0

    # .....mavic_mini_stable/stable_water/1_2_m_only_water.....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/1_2_m_only_water/"
    # video_name = "DJI_0184"
    # video_format = ".MP4"
    # mag_thresh = 1.0

    # .....mavic_mini/mavic_mini_stable/non_water/0_2.....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/non_water/0_2/"
    # video_name = "DJI_0163"
    # video_format = ".MP4"
    # mag_thresh = 1.0

    #.....mavic_mini_stable/stable_water/2_4_m.....NA
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/2_4_m/"
    # video_name = "DJI_0116"
    # video_format = ".MP4"
    # mag_thresh = 1.0

    #.....mavic_mini_stable/stable_water/2_4_m_only_water.....NA
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/2_4_m_only_water/"
    video_name = "DJI_0115"
    video_format = ".MP4"
    mag_thresh = 0.5

    #.....Phantom.....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/phantom/down/non_rain/"
    # video_name = "DJI_0004"
    # video_format = ".MOV"
    # mag_thresh = 0.5
    
    video_format_save_as = ".MP4"
    video = video_path + video_name + video_format
    result_video_save_as = video_path + "result_wd_" + video_name + video_format_save_as
    result_image_save_as = video_path

    frame_array, fps, mag_max, mag_min = view_dense_potical_flow_static(video_path=video,seconds_to_skip=0,resize_factor=2,save_path = result_image_save_as, mag_thresh=mag_thresh)

    # save_video(frame_array,fps = fps ,save_path = result_video_save_as)

