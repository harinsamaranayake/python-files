#https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

import numpy as np
import cv2 as cv

def skip_frames(cap,frame_rate,seconds_to_skip):
    # skip frames
    print('wait! skipping frames')
    frame_rate = frame_rate
    seconds_to_skip = seconds_to_skip

    for i in range(frame_rate*seconds_to_skip):
        ret, frame = cap.read()
        # print('skipped frame:\t', i)

    return cap

def view_dense_potical_flow(video_path=None):
    cap = cv.VideoCapture(cv.samples.findFile(video_path))
    cap = skip_frames(cap=cap,frame_rate=30,seconds_to_skip=10)
    ret, frame1 = cap.read()
    frame1 = cv.resize(frame1, (int(frame1.shape[1]/2), int(frame1.shape[0]/2)))

    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    # set all the values of column 1 to 255
    # hsv[:, 1] = 255
    # hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        frame2 = cv.resize(frame2, (int(frame2.shape[1]/2), int(frame2.shape[0]/2)))
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
        bgr = cv.cvtColor(hsv,cv.COLOR_BGR2GRAY)

        cv.imshow('frame',frame2)
        cv.imshow('bgr',bgr)
        
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

if __name__ == "__main__":
    video_name= "DJI_0004.MOV" #"DJI_0112_S_1.MOV" #"DJI_0002_S_1.MOV" #"DJI_0004.MOV" #"DJI_0002_S_1.MOV" #"DJI_0010.MOV"
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/down/"+video_name
    view_dense_potical_flow(video_path=video_path)

