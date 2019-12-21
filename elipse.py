#Link > https://stackoverflow.com/questions/42206042/ellipse-detection-in-opencv-python
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import cv2
import numpy as np
from PIL import Image

image_path = '/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/DJI_0004_00000000.png'

# Load picture, convert to grayscale and detect edges | using skimage
# image_rgb = data.coffee()[0:220, 160:420]
# image_gray = color.rgb2gray(image_rgb)
# print(image_gray .shape)
# edges = canny(image_gray, sigma=2.0,low_threshold=0.55, high_threshold=0.8)
# img = Image.fromarray(edges)
# img.show()

image_rgb = cv2.imread(image_path)
print(image_rgb.shape)

image_rgb=cv2.resize(image_rgb,(960,540))
print(image_rgb.shape)

image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
print(image_gray .shape)


edges=cv2.Canny(image=image_gray, threshold1=10, threshold2=30)
print(edges.shape)

edges=cv2.GaussianBlur(edges,(3,3),sigmaX=2.0,sigmaY=2.0)

# cv2.imshow('edges',edges)
# cv2.waitKey(0)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
# result = hough_ellipse(edges, accuracy=20, threshold=250,min_size=100, max_size=120)
result = hough_ellipse(edges, accuracy=1, threshold=150,min_size=1, max_size=270)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
