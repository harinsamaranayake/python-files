# Link > https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
# Link > https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghcircles.html

import cv2
import numpy as np

img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/video2frames/DJI_0004_00000540.png"
video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/DJI_0004.MOV"

frame_array=[]

def display_image(flag=0, frame=None):
    # flag 0 - read image from path, flag 1 - image is passed
    img = None
    cimg = None

    if (flag == 0):
        # cv2.imread(img_path, 0) ; 0 - gray image
        img = cv2.imread(img_path, 0)
    else:
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.medianBlur(img, 5)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(
        img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100)
        # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=40, maxRadius=100)

    #np.around() > EVENLY round to the given number of decimals.
    #np.unit16 > Unsigned integer (0 to 65535)
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    if (flag == 0):
        #single image
        cv2.imshow('detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return cimg


def display_video():
    cap = cv2.VideoCapture(video_path, 0)
    while(True):
        ret, frame = cap.read()
        if ret:
            output_frame = display_image(1, frame)
            # cv2.imshow('output', output_frame)
            frame_array.append(output_frame)
        else:
            break
        # cv2.waitKey(x) x-delay x miliseconds x=0-infinite dealy
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def get_video():
    #https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    #https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html

    # out = cv2.VideoWriter(filename='project.avi',fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=100,isColor=1)
    out = cv2.VideoWriter(filename='result_video.avi',fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=(960,540),isColor=1)
 
    for i in range(len(frame_array)):
        out.write(frame_array[i])
        print(i)

    out.release()


if __name__ == "__main__":
    # display_image()
    display_video()
    get_video()
    print('done!')
