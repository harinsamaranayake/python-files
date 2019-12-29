import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import KernelDensity

img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/DJI_0004_00000000.png"

img = mpimg.imread(img_path)
img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
print(img.shape)

img_cv = cv2.imread(img_path,1)
img_cv = cv2.resize(img_cv, (int(img_cv.shape[1]/2), int(img_cv.shape[0]/2)))
img_cv = cv2.medianBlur(img_cv, 3)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

circles = cv2.HoughCircles(
    img_cv, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100)
    # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=40, maxRadius=100)

if type(circles) is np.ndarray:
    # np.around() > EVENLY round to the given number of decimals.
    # np.unit16 > Unsigned integer (0 to 65535)
    circles = np.uint16(np.around(circles))
    circles_xyr = circles[0, :]
    circles_xy = circles_xyr[:, 0:2]
    circle_x = circles_xyr[:, 0:1]
    circle_y = circles_xyr[:, 1:2]

    print(circles_xy)

# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
# single point marker
# plt.scatter([25,25], [50,100], s=10, c='red', marker='o')

kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
# kde = KernelDensity(bandwidth=0.03, metric='haversine')

kde.fit(circles_xy)

levels = np.linspace(0, 2, 25)
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contour.html
plt.contourf(circle_x, circle_y, Z, levels=levels, cmap='Purples')

imgplot = plt.imshow(img)

plt.show()