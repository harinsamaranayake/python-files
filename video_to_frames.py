import cv2
read_path="/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/"
write_path="/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/video2frames/"
video_name="DJI_0004"
video_extention=".MOV"
image_format=".png"

# read_path="/Users/harinsamaranayake/Documents/Research/Datasets/car_videos/02/"
# write_path="/Users/harinsamaranayake/Documents/Research/Datasets/car_images/"
# video_name="MVI_3165"
# video_extention=".MP4"
# image_format=".png"


vidcap = cv2.VideoCapture(read_path+video_name+video_extention)
success, image = vidcap.read()
frame_count = 0
write_count = 0

while success:
    if(frame_count % 60 == 0):
        # save frame
        print(write_path+"car_"+video_name+("_%s"% str(frame_count).zfill(8))+image_format )
        cv2.imwrite(write_path+"car_"+video_name+("_%s"% str(frame_count).zfill(8))+image_format, image)
        write_count += 1
    success, image = vidcap.read()
    # print('Read a new frame:%d ' % frame_count, success, 'Wrote frame:', write_count)
    frame_count += 1
    # break

print("Write Count > ",write_count)
