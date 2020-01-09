# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.stats as st
import cv2

img_name = "DJI_0004_00000000"
# img_name = "DJI_0004_00000300"
# img_name = "DJI_0004_00000600"
# img_name = "DJI_0004_00000900"
img_extention = ".png"
img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/" + img_name + img_extention

write_path_1 = "/Users/harinsamaranayake/Desktop/" + img_name + "_" + "original" + img_extention
write_path_2 = "/Users/harinsamaranayake/Desktop/" + img_name + "_" + "circles" + img_extention
write_path_3 = "/Users/harinsamaranayake/Desktop/" + img_name + "_" + "centers" + img_extention
write_path_4 = "/Users/harinsamaranayake/Desktop/" + img_name + "_" + "kde_2D" + img_extention
write_path_5 = "/Users/harinsamaranayake/Desktop/" + img_name + "_" + "kde_3D" + img_extention

img = cv2.imread(img_path, 1)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
img_copy = img.copy()
img = cv2.medianBlur(img, 3)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# print(img.shape)

cimg_outer = img_copy.copy()
cimg_inner = img_copy.copy()

circles = cv2.HoughCircles(
    img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100)

if type(circles) is np.ndarray:
    circles = np.uint16(np.around(circles))
    circles_xyr = circles[0, :]
    circles_xy = circles_xyr[:, 0:2]
    circles_x = circles_xyr[:, 0:1]
    circles_y = circles_xyr[:, 1:2]
    circles_z = circles_xyr[:, 2:3]
    # print(circles_xy)

    for i in circles[0, :]:
        # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
        # cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)

        # draw the outer circle
        cv2.circle(cimg_outer, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(cimg_inner, (i[0], i[1]), 2, (0, 0, 255), 3)

x = circles_x.ravel()
y = circles_y.ravel()
w = circles_z.ravel()

# Note : Top left coordinate of an image is 0,0. But in plt 0,0 is the bottom left coordinate.
# To achive this following inversion is applied.
xmin, xmax = 0, img.shape[1]
ymin, ymax = img.shape[0], 0
# print(w)

#.................KDE...................#

# -1 | to reduce by 1
xx, yy = np.mgrid[xmin:xmax:1, ymin:ymax:-1]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])

# kernel = st.gaussian_kde(dataset=values,bw_method=None,weights=None)
kernel = st.gaussian_kde(dataset=values,bw_method='silverman',weights=w) # scott silverman scalar(0.2)

f = np.reshape(kernel(positions).T, xx.shape)
print(f.shape)

#..............Figure 01................#

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Contourf plot | Filled
cfset = ax.contourf(xx, yy, f, cmap='Blues')

# Contour plot | Lines
# cset = ax.contour(xx, yy, f, colors='k')

# Or kernel density estimate plot instead of the contourf plot
# ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

# Label plot
# ax.clabel(cset, inline=10, fontsize=10)

ax.set_xlabel('Y')
ax.set_ylabel('X')

plt.axis('scaled')
# plt.savefig(write_path_4)
plt.show()

#..............Figure 02................#

fig = plt.figure(figsize=(10,5))
# fig.suptitle('Gaussian KDE', fontsize=12)
ax = fig.add_subplot(111, projection='3d')

# Data
X = np.arange(xmin, xmax, 1)
Y = np.arange(ymax, ymin, 1)
X, Y = np.meshgrid(X, Y)
Z = f.T
print('x',X.shape)
print('y',Y.shape)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',linewidth=1, antialiased=False)

# Customize the z axis.
z_max = np.amax(f)
ax.set_zlim(0, z_max)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=1, aspect=2)

plt.savefig(write_path_5)
plt.show()

#...............View cv2................#

# cv2.imshow('img_copy',img_copy)
# cv2.imshow('cimg_outer',cimg_outer)
# cv2.imshow('cimg_inner',cimg_inner)
# cv2.waitKey(0)

# cv2.imwrite(write_path_1, img_copy)
# cv2.imwrite(write_path_2, cimg_outer)
# cv2.imwrite(write_path_3, cimg_inner)