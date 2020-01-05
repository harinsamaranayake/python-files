# Link > https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
# Link > https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghcircles.html

import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

frame_array = []
cimg_temp_array = []

# Image size reduce factor
resize_factor = 2


def get_empty_image(height, width):
    cimg_empty = np.zeros((height, width, 3), np.uint8)
    return cimg_empty


def display_image_m1(flag=0, frame=None):
    # hough circles
    # flag 0 - read image from path, flag 1 - read passed image
    original = None

    img = None
    cimg = None

    if (flag == 0):
        # cv2.imread(img_path, 0) | 0 - gray image , 1 - color
        img = cv2.imread(img_path, 0)
    else:
        img = frame

    
    img = cv2.resize(
        img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))
    original = img
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    

    # output_frame | background gray and circles color
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
    # param1 | In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny()
    # edge detector (the lower one is twice smaller)
    # param2 | In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage.
    # The smaller it is, the more false circles may be detected.
    # return type | type(circles) | if no circles detected > 'NoneType', if circles detected > 'numpy.ndarray'
    circles = cv2.HoughCircles(
        # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100) # 2
        # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=50, minRadius=60, maxRadius=100)
        img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=50, minRadius=50, maxRadius=120)
        # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=40, maxRadius=100)
        # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=36, param2=30, minRadius=30, maxRadius=50)

    if type(circles) is np.ndarray:
        # np.around() > EVENLY round to the given number of decimals.
        # np.unit16 > Unsigned integer (0 to 65535)
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
            # cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)

            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    if (flag == 0):
        # single image
        cv2.imshow('hough_circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return original, cimg, cimg, cimg


def display_image_m2(flag=0, frame=None):
    # flag 0 - read image from path, flag 1 - read passed image
    original = None
    img = None
    cimg = None
    cimg_map = None
    cimg_map_norm = None
    heat_map = None

    if (flag == 0):
        # cv2.imread(img_path, 0) | 0 - gray image , 1 - color image
        img = cv2.imread(img_path, 1)
    else:
        img = frame

    print(img.shape)

    img = cv2.resize(
        img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))
    original = img
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cimg = img

    circles = cv2.HoughCircles(
        img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100)
    # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=40, maxRadius=100)

    if type(circles) is np.ndarray:
        # np.around() > EVENLY round to the given number of decimals.
        # np.unit16 > Unsigned integer (0 to 65535)
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            print(i)
            # creater empty image
            cimg_temp = np.zeros_like(img)

            # draw the outer circle | -1 - filled image
            cv2.circle(cimg_temp, (i[0], i[1]), i[2], (2, 2, 2), -1)

            # draw the center of the circle
            # cv2.circle(cimg_temp, (i[0], i[1]), 2, (0, 0, 255), 3)

            # collect annotations
            # NOTE : each circle is taken as a separate frame
            cimg_temp_array.append(cimg_temp)

            # display image with a single circle
            # cv2.imshow('single_circle_frame', cimg_temp)
            # cv2.waitKey(100)

        # obtaining composite image of all detected circles
        # cimg_map=np.zeros_like(img)
        # for i in range(len(cimg_temp_array)):
        #     cimg_map=cv2.add(cimg_map,cimg_temp_array[i])
        #     # cv2.imshow('test_4', cimg_map)
        #     # cv2.waitKey(100)

        # obtain the mean of the cimg_map array
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        cimg_map = np.mean(cimg_temp_array, axis=0)
        # cimg_map=np.median(cimg_temp_array, axis=0)
        # cimg_map=stats.mode(cimg_temp_array)

        # Normalize the image
        # max_pixel_value = np.max(cimg_map.flatten())
        # min_pixel_value = np.min(cimg_map.flatten())
        # cimg_map = 255 * (cimg_map / max_pixel_value)
        # faster method :
        # cimg_map_norm = None
        cimg_map_norm = cv2.normalize(
            cimg_map, cimg_map_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # get heat map
        heat_map = get_heat_map(cimg_map_norm)

    if (flag == 0):
        # image
        image_type = cimg_map  # heat_map
        cv2.imshow('single_image_cmap', image_type)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save image
        # Rescale the image to 255
        # image_type = cv2.convertScaleAbs(image_type, alpha=(255.0))
        # cv2.imwrite('/Users/harinsamaranayake/Desktop/single_image_display.jpg',image_type)

    else:
        # video
        return original, cimg_map, cimg_map_norm, heat_map

def display_image_m3_kde(flag=0, frame=None):
    # flag 0 - read image from path, flag 1 - read passed image
    original = None
    img = None
    cimg = None
    cimg_map = None
    cimg_map_norm = None
    heat_map = None
    kde_img = None

    if (flag == 0):
        # cv2.imread(img_path, 0) | 0 - gray image , 1 - color image
        img = cv2.imread(img_path, 1)
    else:
        img = frame

    print(img.shape)

    img = cv2.resize(
        img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))
    original = img
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cimg = img

    circles = cv2.HoughCircles(
        img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=60, maxRadius=100)
        # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=30, minRadius=40, maxRadius=100)

    if type(circles) is np.ndarray:
        # np.around() > EVENLY round to the given number of decimals.
        # np.unit16 > Unsigned integer (0 to 65535)
        circles = np.uint16(np.around(circles))
        circles_xyr = circles[0, :]
        circles_xy = circles_xyr[:, 0:2]

        print(circles_xy)

        # for i in circles[0, :]:
        #     # print('x:\t',i[0],'\ty:\t',i[1],'\tr:\t',i[2])
        #     pass

        # for i in circles_xy:
        #     print(i)
        #     pass

        # create empty image
        kde_img = np.zeros_like(img)

        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit()

            

            

        

    if (flag == 0):
        # image
        image_type = original
        cv2.imshow('single_image_original', image_type)
        cv2.imshow('kde_img', kde_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save image
        # Rescale the image to 255
        # image_type = cv2.convertScaleAbs(image_type, alpha=(255.0))
        # cv2.imwrite('/Users/harinsamaranayake/Desktop/single_image_display.jpg',image_type)

    else:
        # video
        return original, cimg_map, cimg_map_norm, heat_map


def display_video(video_path=None,frame_rate=30, seconds_to_skip=0):
    cap = cv2.VideoCapture(video_path, 0)

    # skip frames
    skip_frames(cap=cap, frame_rate=30, seconds_to_skip=0)

    while(True):
        ret, frame = cap.read()

        if ret:
            img, cimg_map, cimg_map_norm, heat_map = display_image_m1(1, frame)
            frame_array.append(heat_map)
            
            cv2.imshow('input', img)
            cv2.imshow('cimg_map_maen', cimg_map)
            cv2.imshow('cimg_map_norm', cimg_map_norm)
            cv2.imshow('heat_map', heat_map)
            # cv2.waitKey(0)

            # cv2.imwrite('/Users/harinsamaranayake/Desktop/original.png',img)
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/cimg_map_mean.png',cimg_map)
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/cimg_map_mean_norm.png',cimg_map_norm)
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/heat_map.png',heat_map)
        else:
            break

        # cv2.waitKey(x) x-delay x miliseconds x=0-infinite dealy
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # break


def get_video():
    # https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html

    # out = cv2.VideoWriter(filename='project.avi',fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=100,isColor=1)
    out = cv2.VideoWriter(filename='result_video.avi', fourcc=cv2.VideoWriter_fourcc(
        *'DIVX'), fps=30, frameSize=(960, 540), isColor=1)

    for i in range(len(frame_array)):
        out.write(frame_array[i])
        print('wrote_to_video:\t', i)

    out.release()


def get_heat_map(img):
    # https://stackoverflow.com/questions/56275515/visualizing-a-heatmap-matrix-on-to-an-image-in-opencv
    heatmap = None
    heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Note: HeatMap MatPlotLib | https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    # plt.imshow(cimg_map, cmap='viridis')
    # plt.colorbar()
    # plt.show()

    return heatmap


def skip_frames(cap, frame_rate, seconds_to_skip):
    # skip frames
    print('wait! skipping frames')
    frame_rate = frame_rate
    seconds_to_skip = seconds_to_skip

    for i in range(frame_rate*seconds_to_skip):
        ret, frame = cap.read()
        # print('skipped frame:\t', i)

    return cap


if __name__ == "__main__":
    video_name = "DJI_0004.MOV"  # "DJI_0002_S_1.MOV"  # "DJI_0010.MOV"
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_videos/down/"+video_name
    img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/DJI_0004_00000000.png"
    
    # display_image_m3_kde(flag=0)
    display_video(video_path=video_path,frame_rate=30, seconds_to_skip=0)
    # get_video()
    print('done!')
