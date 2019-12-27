#https://stackoverflow.com/questions/56275515/visualizing-a-heatmap-matrix-on-to-an-image-in-opencv

import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

img_path = "/Users/harinsamaranayake/Desktop/test.jpg"
img = cv2.imread(img_path,1)

def get_heat_map(img):
    # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
    # https://stackoverflow.com/questions/38025838/normalizing-images-in-opencv/38041997
    heatmap = None
    heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # https://docs.opencv.org/2.4/modules/contrib/doc/facerec/colormaps.html
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap
    
if __name__ == "__main__":
    img = cv2.imread(img_path,1)
    heatmap = get_heat_map(img)
    cv2.imshow("Heatmap", heatmap)
    cv2.waitKey(0)