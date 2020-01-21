import cv2
import numpy as np
import keras
import skimage.io as io
import skimage.transform as trans

path = "/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/crop/test_pred/5_predict.png"

for i in range(30):
    path_new = path + str(i) + "_predict.png"
    print(path_new)
    img = cv2.imread(path)
    print(np.amax(img),'\t',np.amin(img))

