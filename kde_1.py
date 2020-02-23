import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plot
import cv2
from sklearn.neighbors import KernelDensity

def kde_scipy( vals1, vals2, a, b, c, d, xN,yN ):
    #vals1, vals2 are the values of two variables (columns)
    #(a,b) interval for vals1; usually larger than (np.min(vals1), np.max(vals1))
    #(c,d) -"-          vals2 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html

    x=np.linspace(a,b,xN)
    y=np.linspace(c,d,yN)
    X,Y=np.meshgrid(x,y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([vals1, vals2])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    print(values)

    return [x, y, Z]

img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_03_down/DJI_0004_00000000.png"
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
    print(circles_xy)

vals1=circles_x.ravel()
vals2=circles_y.ravel()
x0=0
xn=img.shape[1]
y0=0
yn=img.shape[0]
xN=img.shape[1]
yN=img.shape[0]
# print(vals1)
# print(vals2)

x, y, Z = kde_scipy( vals1=vals1, vals2=vals2, a=x0, b=xn, c=y0, d=yn, xN=xN, yN=yN )
levels = np.linspace(0, Z.max(), 10)

# fig, ax = plot.subplots(1, 2)
# ax[1].contourf(x, y, Z, levels=levels, cmap='Purples')
# ax[0].imshow(img)

# plot.scatter(vals1, vals2, s=10, c='red', marker='o')
plot.contourf(x, y, Z, levels=levels, cmap='Purples')

# plot.axis('equal')
plot.axis('scaled')

plot.show()