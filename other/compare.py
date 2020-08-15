import cv2
import numpy as np

img=cv2.imread('img_000000101_original.png')
img=cv2.resize(img,(640,360))
shape=img.shape
print(shape)
cv2.imshow('original',img)

img2=cv2.imread('img_000000101_result.png')
img2=cv2.resize(img2,(640,360))
shape2=img2.shape
print(shape2)
cv2.imshow('result',img2)

cv2.waitKey(0)