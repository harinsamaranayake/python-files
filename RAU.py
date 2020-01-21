import numpy as np
import cv2
import os
import tensorflow as tf
from keras.layers import *

def RA_unit(x, h, w, n):
    print("\nx, h, w and n :", x.shape, h, w, n,'\n')
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    print('\nx_1\t',np.shape(x_1),'\n')
    print('\nx_t\t',np.shape(x_t),'\n')
    for k in range(n):
        x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,int(w/2),x.shape[3].value])
        x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1)
        x_t_3 = tf.abs(x - x_t_2)
        x_t = tf.concat([x_t, x_t_3], axis=3)
        print('\nx_t_1\t',np.shape(x_t_1),'\n')
        print('\nx_t_2\t',np.shape(x_t_2),'\n')
        print('\nx_t_3\t',np.shape(x_t_3),'\n')
        print('\nx_t\t'  ,np.shape(x_t)  ,'\n')
    x_out = tf.concat([x, x_t], axis=3) # [h,w,c(n+1))] | 3[8+1] = 27
    conv = Conv2D(x.shape[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x_out)
    print('\nx_out\t',np.shape(x_out),'\n')
    print('\nconv\t',np.shape(conv),type(x),'\n')
    print('\nx_in\t',np.shape(x),type(x),'\n')
    # conv_p = tf.Session().run(conv)
    # print('\nconv_p\t',conv_p,'\n')
    x_p = tf.Session().run(x)
    print('\nx_p\t',x_p,'\n')
    return conv

def RA_unit_original(x, h, w, n):
    print("\nx, h, w and n :", x.shape, h, w, n,'\n')
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    x_2 = tf.zeros([1, h, w, 0], tf.float32)
    x_t_small = tf.zeros([1, x_1.shape[1].value, w/2, 0], tf.float32)
    x_out = None
    for k in range(n):
        x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,int(w/2),x.shape[3].value])
        x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1)
        x_2 = tf.concat([x_2, x_t_2], axis=3)
        x_t_3 = tf.abs(x - x_t_2)
        x_t = tf.concat([x_t, x_t_3], axis=3)
    x_out = tf.concat([x, x_t], axis=3)
    print('\nx_out\t',np.shape(x_out),'\n')
    return x_out

def RA_unit_new(x, h, w, n):
    # x-input, h-height, w-width, n-slice count
    print("\nx, h, w and n :", x.shape, h, w, n,'\n')

    # if the image height is h and required slice count is n, then the slice height is h/n. So the stride is also h/n.
    # https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool
    # tf.nn.avg_pool(input,ksize,strides,padding,data_format=None,name=None)
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1],strides=[1, h/n, 2, 1], padding="SAME")

    x_t = tf.zeros([1, h, w, 0], tf.float32)

    for k in range(n):
        x_t_1 = tf.slice(x_1, [0, k, 0, 0], [1, 1, int(w/2), int(x.shape[3].value)]) # tf.slice(input,begin,size,name=None)
        x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1) # resize back to h,w
        x_t_3 = tf.abs(x - x_t_2) # i'-x'
        x_t = tf.concat([x_t, x_t_3], axis=3) # concatenating to obtain D = I' - X' = k(i'-x')
        
        # print('x_t_2\n',x_t_2[0],'\n')
        # print('x\n',x[0],'\n')

        x_t_2 = tf.Session().run(x_t_2)
        x = tf.Session().run(x)

        # print('x_t_2\n',x_t_2[0],'\n')
        # print('x\n',x[0],'\n')

        x_t_2 = x_t_2.astype(np.uint8) #'int8'
        x = x.astype(np.uint8)

        # print(type(x_t_2), x_t_2.dtype, np.shape(x_t_2[0]))
        # print(type(x), x.dtype, np.shape(x[0]))

        # print('x_t_2\n',x_t_2[0],'\n')
        # print('x\n',x[0],'\n')

        # cv2.imshow('x_t_2', x_t_2[0])
        # cv2.imshow('x', x[0])
        # cv2.waitKey(0)

        # print('x_t_1\t',np.shape(x_t_1))
        # print('x_t_2\t',np.shape(x_t_2))
        # print('x_t_3\t',np.shape(x_t_3))
        # print('x_t\t',np.shape(x_t),'\n')

        break

    x_out = tf.concat([x, x_t], axis=3)  # I + D
    print('\nx_out\t',np.shape(x_out),'\n')

    return x_out

if __name__ == "__main__":
    image_name = "img_000000101.png"
    image_path = "/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color/"+image_name
    img = cv2.imread(image_path)
    # print(type(img), np.shape(img))
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    img = img.reshape([1,img.shape[0],img.shape[1],img.shape[2]])
    # print(type(img), np.shape(img))
    img = tf.cast(img, tf.float32)
    # print(type(img), np.shape(img))

    # n = int(8)
    # h = int(img.shape[1])
    # w = int(img.shape[2])
    # z = int(h/n)
    # print('z',z,h,w,n)

    # resize_factor = 2
    # img = cv2.resize(img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)))

    x_out = RA_unit(x = img, h = img.shape[1].value, w =img.shape[2].value, n = 8)