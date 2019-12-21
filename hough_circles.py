# Link > https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
# Link > https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghcircles.html

import cv2
import numpy as np

img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/video2frames/DJI_0004_00000540.png"
video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/DJI_0004.MOV"


def display_image(flag=0, frame=None):
    img = None
    cimg = None

    if (flag == 0):
        # 0 - gray image
        img = cv2.imread(img_path, 0)
    else:
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.medianBlur(img, 5)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(
        img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    if (flag == 0):
        cv2.imshow('detected circles', cimg)
        cv2.waitKey(0)
        print('completed')
        cv2.destroyAllWindows()
    else:
        return cimg


def display_video():
    cap = cv2.VideoCapture(video_path, 0)
    while(True):
        ret, frame = cap.read()
        output_frame = display_image(1, frame)
        cv2.imshow('output', output_frame)
        # cv2.waitKey(x) x-delay x miliseconds x=0-infinite
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # display_image()
    display_video()
