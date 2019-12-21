# Note Dont keep the script within the image folder
import cv2
import os
original_path='/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/'
mask_path='/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_gt/'
original_img_list = os.listdir(original_path)
mask_img_list = os.listdir(mask_path)
count=0

# print(original_img_list)
# print(mask_img_list)

for i in original_img_list:
    count+=1
    if i in mask_img_list:
        pass 
    else:
        print(i)

print(count)
