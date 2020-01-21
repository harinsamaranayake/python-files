import numpy as np
import cv2
import os

def get_cropped_images(path_color=None, path_mask=None):
    path_color = path_color
    path_mask = path_mask

    print('paths:\t', path_color, '\t', path_mask)

    # File name at [2]
    color_name_list = next(os.walk(path_color))[2]

    # Remove '.DS_Store'
    if '.DS_Store' in color_name_list:
        color_name_list.remove('.DS_Store')

    color_img_list = []
    mask_img_list = []

    crop_height = 256
    crop_width = 256

    for name in color_name_list:
        # to obtain predicted and gt images 0 - gray | 1 - color
        color_img = cv2.imread(path_color+"/%s" % name)
        mask_img = cv2.imread(path_mask+"/%s" % name)

        # Matching both the color and the mask images
        # print(color_img.shape)
        # print(mask_img.shape)
        # color_img = cv2.resize(color_img, (mask_img.shape[1], mask_img.shape[0]))
        mask_img = cv2.resize(
            mask_img, (color_img.shape[1], color_img.shape[0]))
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
            y = 0
            x = 0
            ih = image_height
            iw = image_width
            ch = crop_height
            cw = crop_width

            color_crop_top_left = color_img[y:y+ch, x:x+cw]
            mask_crop_top_left = mask_img[y:y+ch, x:x+cw]

            color_crop_top_right = color_img[y:y+ch, iw-cw:iw]
            mask_crop_top_right = mask_img[y:y+ch, iw-cw:iw]

            color_crop_bottom_left = color_img[ih-ch:ih, x:x+cw]
            mask_crop_bottom_left = mask_img[ih-ch:ih, x:x+cw]

            color_crop_bottom_right = color_img[ih-ch:ih, iw-cw:iw]
            mask_crop_bottom_right = mask_img[ih-ch:ih, iw-cw:iw]

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

    # converting to numpy arrays
    color_img_array = np.array(color_img_list)
    mask_img_array = np.array(mask_img_list)

    return color_img_array,


def get_resized_images(path_color=None, path_mask=None):
    path_color = path_color
    path_mask = path_mask

    print('paths:\t', path_color, '\t', path_mask)

    # File name at [2]
    color_name_list = next(os.walk(path_color))[2]

    # Remove '.DS_Store'
    if '.DS_Store' in color_name_list:
        color_name_list.remove('.DS_Store')

    color_img_list = []
    mask_img_list = []

    resize_height = 256
    resize_width = 256

    for name in color_name_list:
        # to obtain predicted and gt images 0 - gray | 1 - color
        color_img = cv2.imread(path_color+"/%s" % name)
        mask_img = cv2.imread(path_mask+"/%s" % name)

        # Matching both the color and the mask images | 0-height 1-width
        # print(color_img.shape)
        # print(mask_img.shape)
        color_img = cv2.resize(color_img, (resize_width, resize_height))
        mask_img = cv2.resize(mask_img, (resize_width, resize_height))
        # print(color_img.shape)
        # print(mask_img.shape)

        # Change Format
        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # Threshold images
        # ret_1,color_img = cv2.threshold(color_img,128,255,cv2.THRESH_BINARY)
        ret_2, mask_img = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)

        # View images
        # cv2.imshow('color_img',color_img)
        # cv2.imshow('mask_img',mask_img)
        # cv2.waitKey(0)

        color_img_list.append(color_img)
        mask_img_list.append(mask_img)

    # converting to numpy arrays
    color_img_array = np.array(color_img_list)
    mask_img_array = np.array(mask_img_list)

    return color_img_array, mask_img_array


def save_image(img=None, img_name=None, save_path=None):
    # if img is an image array
    for i in range(img.shape[0]):
        save_as = save_path + str(i) + ".png"
        # print(save_as)
        # print(img[i].shape)
        # cv2.imshow('test',img[i])
        # cv2.waitKey(0)
        cv2.imwrite(save_path, img[i])

    # if img is an simgle image
    # save_path =  ave_path + "/" + img_name + ".png"
    # v2.imwrite( save_path , img )


if __name__ == "__main__":
    path_color = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/original/color/on_road'
    path_mask = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/original/mask/on_road'
    save_path = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/original/color/on_road/'

    # color_img_array, mask_img_array = get_cropped_images(path_color=path_color,path_mask = path_mask)
    color_img_array, mask_img_array = get_resized_images(path_color=path_color,path_mask = path_mask)

    print(color_img_array.shape,mask_img_array.shape)

    save_image(img=color_img_array,save_path=save_path)