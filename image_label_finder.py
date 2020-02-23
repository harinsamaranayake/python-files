import numpy as np
import cv2
import os

original_image_path='/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color'
label_image_path='/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_gt'

original_list = next(os.walk(original_image_path))[2]
label_list = next(os.walk(label_image_path))[2]

if '.DS_Store' in original_list:
    original_list.remove('.DS_Store')

if '.DS_Store' in label_list:
    label_list.remove('.DS_Store')
    
print('\nnot in label_list')
for img in original_list:
    if(img in label_list):
        pass
    else:
        print(img)
print('completed')

print('\nnot in original_list')
for img in label_list:
    if(img in original_list):
        pass
    else:
        print(img)
print('completed')