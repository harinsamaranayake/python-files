import numpy as np
import cv2
import time
from PIL import Image


def view_hsv(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    cv2.imshow('hsv_img', hsv_img)
    cv2.imshow('h', h)
    cv2.imshow('s', s)
    cv2.imshow('v', v)


def canny_threshold(src, src_gray, low_threshold=50, ratio=3, kernel_size=3):
    img_blur = cv2.blur(src_gray, (3, 3))
    detected_edges = cv2.Canny(
        img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:, :, None].astype(src.dtype))
    cv2.imshow("t"+str(low_threshold)+"_r" +
               str(ratio)+"_k"+str(kernel_size), dst)
    return dst


def dilate_erode(img):
    kernel_dilate = np.ones((3, 3), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)
    itr = 1

    # erode
    erosion = cv2.erode(img, kernel_erode, iterations=itr)

    # dilation
    dilation = cv2.dilate(img, kernel_dilate, iterations=itr)

    # dilation_erosion
    dilation_erosion = cv2.erode(dilation, kernel_erode, iterations=itr)

    # erosion_dilation
    erosion_dilation = cv2.erode(erosion, kernel_dilate, iterations=itr)

    cv2.imshow('erode', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('dilation_erosion', dilation_erosion)
    cv2.imshow('erosion_dilation', erosion_dilation)


def patch_threhold(frame, blocksize):
    frame = cv2.resize(frame, (960, 540))

    print(frame.shape)
    height = frame.shape[0]
    width = frame.shape[1]

    # creating a new image with block average
    if((width % blocksize == 0) & (height % blocksize == 0)):
        new_img = [[0 for i in range(int(width/blocksize))]
                   for i in range(int(height/blocksize))]

        new_img = array = np.array(new_img, dtype=np.uint8)
        # print(new_img.shape)

        for x in range(0, width, blocksize):
            for y in range(0, height, blocksize):
                block_sum = 0
                block_sum_average = 0
                # taking sum within a block
                # range(0,blocksize) = 0 to blocksize-1
                for p in range(0, blocksize):
                    for q in range(0, blocksize):
                        block_sum = block_sum+frame[y+p][x+q]
                block_sum_average = int(round(block_sum/(blocksize*blocksize)))
                new_img[round(y/blocksize)][round(x/blocksize)
                                            ] = block_sum_average
                print(str(y)+" "+str(x)+" "+str(block_sum) +
                      " "+str(block_sum_average))

        cv2.imshow("test", new_img)
        cv2.waitKey(0)
        print("done")

    else:
        print("blocksize cant divide the image")
        return


def get_thresh_image(threshold_value,frame):
    ret = None
    # ret,thresh_img = cv2.threshold(frame,threshold_value,255,cv2.THRESH_BINARY)
    # thresh_img = cv2.adaptiveThreshold(frame,threshold_value,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # thresh_img = cv2.adaptiveThreshold(frame,threshold_value,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # ret,thresh_img = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret

def pre_process():

    # input_video
    cap = cv2.VideoCapture('DJI_0004.MOV')
    # cap = cv2.VideoCapture('DJI_0011.MOV')

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Frame resize
        # original size
        # frame=frame 
        # frame=cv2.resize(frame, (480, 270))
        frame = cv2.resize(frame, (960, 540))

        # Convert image
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)

        # Blur image
        # image_to_blur = v
        # blur_img = cv2.GaussianBlur(image_to_blur, (3, 3), 0)
        # frame=blur_img

        # Threshold image
        # threshold_value = 150
        # frame=get_thresh_image(threshold_value,frame)

        # Display the resulting frame
        # cv2.imshow('output_img', frame)
        # view_hsv(frame)
        canny_threshold(frame,v,low_threshold=10,ratio=3,kernel_size = 3)
        # patch_threhold(frame, 20)

        # Delay seconds
        # time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # For one frame
        # break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    pre_process()
    # frame = cv2.imread('1.png', cv2.IMREAD_COLOR)
    # hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv_img)
    # patch_threhold(v, 4)


if __name__ == "__main__":
    main()
