# Harin Samaranayake | Index 15001202 

import cv2

img_1 = cv2.imread('0001.jpg')
img_2 = cv2.imread('0199.jpg')

img_1_gray=cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2_gray=cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

img_1_edges=cv2.Canny(image=img_1_gray, threshold1=10, threshold2=30)
img_2_edges=cv2.Canny(image=img_2_gray, threshold1=10, threshold2=30)

cv2.imshow('img_1_edges',img_1_edges)
cv2.imshow('img_2_edges',img_2_edges)
# cv2.waitKey(0)

cv2.imwrite('img_0001_edges.png',img_1_edges)
cv2.imwrite('img_0199_edges.png',img_2_edges)