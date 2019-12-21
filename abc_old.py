import numpy as np
import cv2

cap = cv2.VideoCapture('DJI_0004.MOV') #'DJI_0004.MOV''DJI_0011.MOV'

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv_img)

    input_img=v

    blur_img = cv2.GaussianBlur(input_img,(3,3),0)
    # input_img=blur_img

    threshold_value=140

    # ret,output_img = cv2.threshold(input_img,threshold_value,255,cv2.THRESH_BINARY)
    # output_img = cv2.adaptiveThreshold(input_img,threshold_value,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # output_img = cv2.adaptiveThreshold(input_img,threshold_value,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # ret,output_img = cv2.threshold(input_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    src=input_img
    low_threshold=50
    ratio=3
    kernel_size=5
    detected_edges = cv2.Canny(input_img, low_threshold, low_threshold*ratio, kernel_size)
    output_img=detected_edges

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(output_img,kernel,iterations = 1)
    # output_img=erosion
    # mask = detected_edges != 0
    # output_img = src * (mask[:,:,None].astype(src.dtype))

    # Display the resulting frame
    output_img=cv2.resize(output_img, (960, 540))
    cv2.imshow('output_img',output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv2.imshow('window_name', dst)