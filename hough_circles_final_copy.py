# Link > https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
# Link > https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghcircles.html

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity
import math

frame_array = []
cimg_temp_array = []

def hough_circles_demo(frame=None, resize_factor=1):
    resize_factor = resize_factor

    img = None
    cimg = None
    original = None

    img = frame

    img = cv2.resize(img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))
    original = img.copy()

    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    method = cv2.HOUGH_GRADIENT
    dp = 1
    minDist = 1
    param1 = 75
    param2 = 90
    minRadius = int((img.shape[1] / 5 )/ 2)
    maxRadius = int(math.sqrt((((img.shape[0] / 2) * (img.shape[0] / 2)) + ((img.shape[1] / 2) * (img.shape[1] / 2)))))

    circles = cv2.HoughCircles(img, method=method, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if type(circles) is np.ndarray:
        circles = np.uint16(np.around(circles))

        circles_xyr = circles[0, :]
        circles_xy = circles_xyr[:, 0:2]

        # create empty image
        kde_img = np.zeros_like(img)

        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit()

        # draw circles
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    return original, cimg

def save_video(frame_array=None, fps=0, save_path=None):
    # https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html

    out = cv2.VideoWriter(filename=save_path, fourcc=cv2.VideoWriter_fourcc(
        *'DIVX'), fps=fps, frameSize=(960, 540), isColor=1)

    frame_array = frame_array

    for i in range(len(frame_array)):
        out.write(frame_array[i])
        print('wrote_to_video:\t', i)

    out.release()

    print("\nVideo Saved")

def skip_frames(cap,seconds_to_skip):
    # skip frames in a video
    print('wait! skipping frames')

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    seconds_to_skip = seconds_to_skip

    for i in range(fps*seconds_to_skip):
        ret, frame = cap.read()

    return cap

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

def display_image_v1(flag=0, frame=None, resize_factor=2):
    # hough circles
    # flag 0 - read image from path, flag 1 - read the passed image
    # this algorithm is designed for 540,960 resolution of the frame
    resize_factor = resize_factor

    img = None
    cimg = None
    original = None

    if (flag == 0):
        # cv2.imread(img_path, 0) | 0 - gray image , 1 - color image
        img = cv2.imread(img_path, 0)
    else:
        img = frame

    img = cv2.resize(
        img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))
    original = img.copy()

    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.shape)

    # output_frame | background gray and circles color
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
    # param1 | In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the
    #   Canny() edge detector (the lower one is twice smaller)
    # param2 | In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage.
    #  The smaller it is, the more false circles may be detected.
    # dp – Inverse ratio of the accumulator resolution to the image resolution. For example,
    # If dp=1 , the accumulator has the same resolution as the input image.
    # If dp=2 , the accumulator has half as big width and height.
    # return type | type(circles) | if no circles detected > 'NoneType', if circles detected > 'numpy.ndarray'

    # method = cv2.HOUGH_GRADIENT
    # dp = 1
    # minDist = 1
    # param1 = 75
    # param2 = 95
    # minRadius = int((img.shape[1] / 5 )/ 2)
    # maxRadius = int(math.sqrt((((img.shape[0] / 2) * (img.shape[0] / 2)) + ((img.shape[1] / 2) * (img.shape[1] / 2)))))

    method = cv2.HOUGH_GRADIENT
    dp = 1
    minDist = 10
    param1 = 75
    param2 = 80
    minRadius = int((img.shape[1] / 3 )/ 2)
    maxRadius = int(math.sqrt((((img.shape[0] / 2) * (img.shape[0] / 2)) + ((img.shape[1] / 2) * (img.shape[1] / 2)))))

    print(minRadius,maxRadius)

    circles = cv2.HoughCircles(
        img, method=method, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=75, param2=50, minRadius=50, maxRadius=120)

    if type(circles) is np.ndarray:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html
        # np.around() > EVENLY round to the given number of decimals.
        # np.unit16 > Unsigned integer (0 to 65535) | casting
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.htmlq
            # cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)

            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    # save image
    same_img=False
    if (same_img):
        save_name = "method=" + str(method) + "_dp=" + str(dp) + "_minDist=" + str(minDist) + "_param1=" + str(param1) + "_param2=" + str(param2) + "_minRadius=" + str(minRadius) + "_maxRadius" + str(maxRadius)
        save_as = '/Users/harinsamaranayake/Desktop/' + save_name + '.png'
        cv2.imwrite(save_as, cimg)

    if (flag == 0):
        # single image
        cv2.imshow('hough_circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        # single image
        image = cimg_map

        # view image
        cv2.imshow('single_image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save image
        # Rescale the image to 255
        # image_type = cv2.convertScaleAbs(image, alpha=(255.0))
        # cv2.imwrite('/Users/harinsamaranayake/Desktop/single_image.jpg',image)

    return original, cimg_map, cimg_map_norm, heat_map

def display_image_m3_kde(flag=0, frame=None,resize_factor=2):
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
        kde.fit(kde_img)

        cv2.imshow('kde_img', kde_img)
        cv2.waitKey(0)

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

def display_image_v1_1(frame=None, resize_factor=None):
    # this algorithm is designed for 540,960 resolution of the frame

    img = None
    cimg = None
    original = None
    resize_factor = resize_factor
    img = frame

    img = cv2.resize(img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))
    original = img.copy()

    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.shape)

    # output_frame | background gray and circles color
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
    # param1 | In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the
    #   Canny() edge detector (the lower one is twice smaller)
    # param2 | In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage.
    #  The smaller it is, the more false circles may be detected.
    # dp – Inverse ratio of the accumulator resolution to the image resolution. For example,
    # If dp=1 , the accumulator has the same resolution as the input image.
    # If dp=2 , the accumulator has half as big width and height.
    # return type | type(circles) | if no circles detected > 'NoneType', if circles detected > 'numpy.ndarray'

    # method = cv2.HOUGH_GRADIENT
    # dp = 1
    # minDist = 1
    # param1 = 75
    # param2 = 95
    # minRadius = int((img.shape[1] / 5 )/ 2)
    # maxRadius = int(math.sqrt((((img.shape[0] / 2) * (img.shape[0] / 2)) + ((img.shape[1] / 2) * (img.shape[1] / 2)))))

    method = cv2.HOUGH_GRADIENT
    dp = 1
    minDist = 10
    param1 = 75
    param2 = 80
    minRadius = int((img.shape[1] / 3 )/ 2)
    maxRadius = int(math.sqrt((((img.shape[0] / 2) * (img.shape[0] / 2)) + ((img.shape[1] / 2) * (img.shape[1] / 2)))))

    print(minRadius,maxRadius)

    circles = cv2.HoughCircles(
        # img, method=method, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=80, param2=50, minRadius=50, maxRadius=120)

    if type(circles) is np.ndarray:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html
        # np.around() > EVENLY round to the given number of decimals.
        # np.unit16 > Unsigned integer (0 to 65535) | casting
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.htmlq
            # cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)

            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    # save image
    same_img=False
    if (same_img):
        save_name = "method=" + str(method) + "_dp=" + str(dp) + "_minDist=" + str(minDist) + "_param1=" + str(param1) + "_param2=" + str(param2) + "_minRadius=" + str(minRadius) + "_maxRadius" + str(maxRadius)
        save_as = '/Users/harinsamaranayake/Desktop/' + save_name + '.png'
        cv2.imwrite(save_as, cimg)

    # if (flag == 0):
    #     # single image
    #     cv2.imshow('hough_circles', cimg)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return original, cimg, cimg, cimg

def display_video(video_path=None, seconds_to_skip=0):
    cap = cv2.VideoCapture(video_path, 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # skip_frames(cap=cap,seconds_to_skip=seconds_to_skip)

    while(True):
        ret, frame = cap.read()

        if ret:
            img, cimg_map, cimg_map_norm, heat_map = display_image_v1_1(frame,2)

            # TYPE of frame to be saved as the video
            frame_array.append(cimg_map)

            # display video
            cv2.imshow('input', img)
            cv2.imshow('cimg_map_maen', cimg_map)
            
            # cv2.imshow('cimg_map_norm', cimg_map_norm)
            # cv2.imshow('heat_map', heat_map)
            # cv2.waitKey(0)

            # .....save the images......
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/original.png',img)
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/cimg_map_mean.png',cimg_map)
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/cimg_map_mean_norm.png',cimg_map_norm)
            # cv2.imwrite('/Users/harinsamaranayake/Desktop/heat_map.png',heat_map)

            # cv2.waitKey(x) , x=1 delay 1 miliseconds , x=0|infinite dealy
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cv2.destroyAllWindows()

    return frame_array,fps

if __name__ == "__main__":
    #.....mavic_mini_stable/stable_water/1_2_m.....
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/1_2_m/"
    video_name = "DJI_1580646956000"
    video_format = ".MP4"

    #.....mavic_mini_stable/stable_water/1_2_m_only_water.....
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/1_2_m_only_water/"
    video_name = "DJI_0113"
    video_format = ".MP4"

    #.....mavic_mini_stable/stable_water/2_4_m.....
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/2_4_m/"
    video_name = "DJI_0112"
    video_format = ".MP4"

    #.....mavic_mini_stable/stable_water/2_4_m_only_water.....
    video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/mavic_mini/mavic_mini_stable/stable_water/2_4_m_only_water/"
    video_name = "DJI_0112"
    video_format = ".MP4"

    #.....Phantom.....
    # video_path = "/Users/harinsamaranayake/Documents/Research/Datasets/new_drone_videos/phantom/down/non_rain/"
    # video_name = "DJI_0004"
    # video_format = ".MOV"
    
    video_format_save_as = ".MP4"
    video = video_path + video_name + video_format
    result_video_save_as = video_path + "result_HC_" + video_name + video_format_save_as
    result_image_save_as = video_path
    
    frame_array, fps = display_video(video_path=video, seconds_to_skip=0)
    # save_video(frame_array,fps = fps ,save_path = result_video_save_as)

    # img_path = "/Users/harinsamaranayake/Documents/Research/Datasets/drone_images/set_02/set_02_color/DJI_0004_00000000.png"
    # display_image_m3_kde(flag=0)

    print('\n.....Process Completed!.....')
