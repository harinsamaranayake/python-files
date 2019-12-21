import numpy as np
import cv2

# file_name='on_road_test.txt'
# image_folder='on_road_test_results'
# image_folder_new='on_road_test_mask_pred'

file_name_path='/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/text/both_road_test.txt'
image_folder_path='/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/masks/both_road'
image_folder_path_new='/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/both_road_test_mask'

p = np.genfromtxt(file_name_path,dtype='str')
print(p.shape[0])

for i in range(p.shape[0]):
    link=p[i][0]
    part=link.split('/')
    image_name=part[4]
    img=cv2.imread(image_folder_path + '/%s' % image_name)
    cv2.imwrite(image_folder_path_new + '/%s' % image_name,img)
    print(image_name)
    # cv2.imshow('img',img)
    # print(img)   