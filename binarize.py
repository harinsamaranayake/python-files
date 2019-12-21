# Note Dont keep the script within the image folder
import cv2
import os
path='/Users/harinsamaranayake/Desktop/test/pred'
arr = os.listdir(path)

for i in arr:
    if (i!='.DS_Store'): 
        print(i)
        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret,thresh_img = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
        cv2.imwrite(i, thresh_img)