# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python

import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as st
import cv2

img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/DJI_0004_00000000.png"
img = cv2.imread(img_path, 1)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
img = cv2.medianBlur(img, 3)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)

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

# data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
# x = data[:, 0]
# y = data[:, 1]
# xmin, xmax = -3, 3
# ymin, ymax = -3, 3

x = circles_x.ravel()
y = circles_y.ravel()
w = circles_z.ravel()

xmin, xmax = 0, img.shape[1]
ymin, ymax = 0, img.shape[0]
print(w)

# Peform the kernel density estimate
# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([xx.ravel(), yy.ravel()])
# values = np.vstack([x, y])
# kernel = st.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, xx.shape)

xx, yy = np.mgrid[xmin:xmax:1, ymin:ymax:1]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])

# kernel = st.gaussian_kde(dataset=values,bw_method=None,weights=None)
kernel = st.gaussian_kde(dataset=values,bw_method='silverman',weights=w) # scott silverman scalar(0.2)

f = np.reshape(kernel(positions).T, xx.shape)
print(f.shape)

fig = pl.figure()
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

pl.axis('scaled')
pl.show()