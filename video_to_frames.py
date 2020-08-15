# Author    :: Harin Samaranayake
# Note      :: Read videos in a folder, and save frames with a given interval
# read_path         - Folder with videos | should specify
# image_save_path   - read_path/images
# label_save_path   - read_path/label  :: Extra
# prefix            - save image prefix

import cv2
import os

read_path="/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water_with_plants/2_4/"

# prefix = "drone_f_"
# prefix = "drone_h_"
prefix = "drone_s_swwp_2_4_"

if not os.path.exists(read_path + "image"):
    try:
        os.makedirs(read_path + "image")
    except FileExistsError:
        print("Folder Exists")
        pass

if not os.path.exists(read_path + "label"):
    try:
        os.makedirs(read_path + "label")
    except FileExistsError:
        print("Folder Exists")
        pass

read_list = next(os.walk(read_path))[2]

for video in read_list:
    name = video.split(".")
    video_name = name[0]
    video_extention = "." + name[1]
    print(name[0],name[1])

    write_path= read_path + "image/"
    image_format=".png"

    interval = 60
    frame_count = 0
    write_count = 0

    vidcap = cv2.VideoCapture(read_path+video_name+video_extention)
    success, image = vidcap.read()

    while success:
        if (frame_count % interval == 0):
            
            # ......save frame......
            print(write_path + prefix + video_name + ("_%s"% str(frame_count).zfill(8))+image_format)
            cv2.imwrite(write_path + prefix + video_name + ("_%s"% str(frame_count).zfill(8))+image_format, image)


            print('Wrote frame:', write_count)
            write_count += 1
            
        success, image = vidcap.read()
        frame_count += 1

    print("Write Count > ",write_count)
