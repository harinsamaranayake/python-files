import numpy as np
import cv2
from scipy import stats

video_name="DJI_0011.MOV" #"DJI_0002_S_1.MOV" #"DJI_0010.MOV"
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
block_size=10

for start_frame in range(total_frames):
    # start_frame  = 0 to total_frames-1
    temp_array=[]
    for i in range(block_size):
        temp_array.append(frame_array[start_frame+i])
        # print(i)
    # cimg_map = np.mean(cimg_temp_array, axis=0)
    temp_array=np.array(temp_array)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
    # mode_frame[0] - [[mode values]] mode_frame[1]- [[mode positions]]
    mode_frame=stats.mode(temp_array,axis=0)
    diff_mode_frame = cv2.subtract(frame_array[start_frame],mode_frame[0][0])

    # mean_frame values are float while frame_array frames are int 
    # mean_frame=np.mean(temp_array, axis=0)
    # mean_frame=mean_frame.astype(np.uint8)
    # dif_mean_frame = cv2.subtract(frame_array[start_frame],mean_frame)

    # print(type(mean_frame),type(frame_array[start_frame]))
    # print(mode_frame[0][0])
    # print(mean_frame)
    # print(frame_array[start_frame])

    mode_frame=mode_frame[0][0]
    
    # cv2.imshow('mode_frame',mode_frame)
    cv2.imshow('dif_mode_frame',diff_mode_frame)
    # cv2.imshow('mean_frame',mean_frame)
    # cv2.imshow('dif_mean_frame',dif_mean_frame)

    #Normalize
    # max_pixel_value = np.max(diff_mode_frame.flatten())
    # print('max_pixel_value',max_pixel_value)
    # diff_mode_frame = 255 * (diff_mode_frame / max_pixel_value)
    # cv2.imshow('dif_mode_frame2',diff_mode_frame)

    diff_mode_frame_norm=None
    diff_mode_frame_norm = cv2.normalize(diff_mode_frame, diff_mode_frame_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('diff_mode_frame_norm',diff_mode_frame_norm)

    heatmap = cv2.applyColorMap(diff_mode_frame_norm, cv2.COLORMAP_JET)
    # cv2.imshow('heat_map',heatmap)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

print('done')

