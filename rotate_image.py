import os
import cv2
import numpy as np

image_folder_path= "/Users/harinsamaranayake/Desktop/on_road"
image_folder_path_new="/Users/harinsamaranayake/Desktop/on_road_rotated_180"
image_list = next(os.walk(image_folder_path))[2]

if '.DS_Store' in image_list:
    image_list.remove('.DS_Store')

for img in image_list:
    original_img = cv2.imread(image_folder_path+"/%s" % img)
    
    h=original_img.shape[0]
    w=original_img.shape[1]
    c=(w/2,h/2)

    M=cv2.getRotationMatrix2D(center=c,angle=180,scale=1.0)
    rotated_img = cv2.warpAffine(original_img,M,dsize=(w,h))

    cv2.imwrite(image_folder_path_new+"/"+img,rotated_img)
    print(img)