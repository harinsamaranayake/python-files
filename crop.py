import numpy as np
import cv2
import os


def get_images(path_color=None, path_mask=None):
    path_color = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color'
    path_mask = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_mask'

    print('paths:\t', path_color, '\t', path_mask)

    # File name at [2]
    color_name_list = next(os.walk(path_color))[2]

    # Remove '.DS_Store'
    if '.DS_Store' in color_name_list:
        color_name_list.remove('.DS_Store')

    color_img_list = []
    mask_img_list = []

    crop_height = 256
    crop_width = 512

    for name in color_name_list:
        # to obtain predicted and gt images
        color_img = cv2.imread(path_color+"/%s" % name)
        mask_img = cv2.imread(path_mask+"/%s" % name)

        # Matching both the color and the mask images
        # print(color_img.shape)
        # print(mask_img.shape)
        # color_img = cv2.resize(color_img, (mask_img.shape[1], mask_img.shape[0]))
        mask_img = cv2.resize(mask_img, (color_img.shape[1], color_img.shape[0]))
        # print(color_img.shape)
        # print(mask_img.shape)

        # Change Format
        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # Threshold images
        # ret_1,color_img = cv2.threshold(color_img,128,255,cv2.THRESH_BINARY)
        ret_2, mask_imgt = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)

        # View images
        # cv2.imshow('color_img',color_img)
        # cv2.imshow('mask_img',mask_img)
        # cv2.waitKey(0)

        # crop images 0-height 1-width
        image_height = color_img.shape[0]
        image_width = color_img.shape[1]

        if((image_height > crop_height) & (image_width > crop_width)):
            # x-0,y-0 top left corner
            y=0
            x=0
            ih=image_height
            iw=image_width
            ch=crop_height
            cw=crop_width

            color_crop_top_left = color_img[y:y+ch, x:x+cw]
            mask_crop_top_left = mask_img[y:y+ch, x:x+cw]

            color_crop_top_right = color_img[y:y+ch, iw-cw:iw]
            mask_crop_top_right= mask_img[y:y+ch, iw-cw:iw]

            color_crop_bottom_left = color_img[ih-ch:ih, x:x+cw]
            mask_crop_bottom_left = mask_img[ih-ch:ih, x:x+cw]

            color_crop_bottom_right = color_img[ih-ch:ih, iw-cw:iw]
            mask_crop_bottom_right= mask_img[ih-ch:ih, iw-cw:iw]

            # cv2.imshow('color_img',color_img)
            # cv2.imshow('color_crop_top_left',color_crop_top_left)
            # cv2.imshow('color_crop_top_right',color_crop_top_right)
            # cv2.imshow('color_crop_bottom_left',color_crop_bottom_left)
            # cv2.imshow('color_crop_bottom_right',color_crop_bottom_right)

            # cv2.imshow('mask_img',mask_img)
            # cv2.imshow('mask_crop_top_left',mask_crop_top_left)
            # cv2.imshow('mask_crop_top_right',mask_crop_top_right)
            # cv2.imshow('mask_crop_bottom_left',mask_crop_bottom_left)
            # cv2.imshow('mask_crop_bottom_right',mask_crop_bottom_right)

            # cv2.waitKey(0)

            color_crop_top_left = np.array(color_crop_top_left)
            color_crop_top_right = np.array(color_crop_top_right)
            color_crop_bottom_left = np.array(color_crop_bottom_left)
            color_crop_bottom_right = np.array(color_crop_bottom_right)

            mask_crop_top_left = np.array(mask_crop_top_left)
            mask_crop_top_right = np.array(mask_crop_top_right)
            mask_crop_bottom_left = np.array(mask_crop_bottom_left)
            mask_crop_bottom_right = np.array(mask_crop_bottom_right)

            color_img_list.append(color_crop_top_left)
            color_img_list.append(color_crop_top_right)
            color_img_list.append(color_crop_bottom_left)
            color_img_list.append(color_crop_bottom_right)

            mask_img_list.append(mask_crop_top_left)
            mask_img_list.append(mask_crop_top_right)
            mask_img_list.append(mask_crop_bottom_left)
            mask_img_list.append(mask_crop_bottom_right)

        break

    # converting to numpy arrays
    color_img_array = np.array(color_img_list)
    mask_img_array = np.array(mask_img_list)

    return color_img_array, mask_img_array


if __name__ == "__main__":
    get_images()
