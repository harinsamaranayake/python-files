import numpy as np
import cv2
from scipy import stats

video_name="DJI_0004.MOV" #"DJI_0002_S_1.MOV" #"DJI_0010.MOV"
video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/down/"+video_name

cap = cv2.VideoCapture(video_path)

frame_array=[]
frame_array_max=100

frame_count = 0

while(True):
    ret,frame = cap.read()
    if(ret==True):
        frame=cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_array.append(frame)
        frame_count+=1
        print('frame_no:\t',frame_count)
    else:
        break
    
    if(len(frame_array)==frame_array_max):
        break
    
print('frame_count:\t',len(frame_array))

total_frames = len(frame_array)
block_size=20

for start_frame in range(total_frames):
    # start_frame  = 0 to total_frames-1
    temp_array=[]
    for i in range(block_size):
        temp_array.append(frame_array[start_frame+i])
        print(i)
    # cimg_map = np.mean(cimg_temp_array, axis=0)
    temp_array=np.array(temp_array)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
    # mode_frame[0] - [[mode values]] mode_frame[1]- [[mode positions]]
    mode_frame=stats.mode(temp_array,axis=0)
    dif_mode_frame = cv2.subtract(frame_array[start_frame],mode_frame[0][0])
    mean_frame=np.mean(temp_array, axis=0)
    # dif_mean_frame = cv2.subtract(frame_array[start_frame],mean_frame)
    # print(type(mean_frame),type(frame_array[start_frame]))
    print(mean_frame)
    print(frame_array[start_frame])

    print(mode_frame[0][0])
    print(frame_array[2])
    cv2.imshow('mode_frame',mode_frame[0][0])
    cv2.imshow('dif_mode_frame',dif_mode_frame)
    cv2.imshow('mean_frame',mean_frame)
    # cv2.imshow('dif_mean',dif_mean_frame)
    cv2.waitKey(0)
    break

print('done')

