# Link - https://github.com/zhixuhao/unet/blob/master/model.py
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import cv2

#Link > https://www.tensorflow.org/guide/gpu
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

image_size = None
image_height = None
image_width = None

def crop_sample(color_img,mask_img):
    pass

def get_images_resized(path_color=None,path_mask=None,height=0,width=0):
    
    print('paths:\t', path_color, '\t', path_mask)

    # File name at [2]
    color_name_list = next(os.walk(path_color))[2]

    # Remove '.DS_Store'
    if '.DS_Store' in color_name_list:
        color_name_list.remove('.DS_Store')

    color_img_list = []
    mask_img_list = []

    for name in color_name_list:
        # to obtain predicted and gt images
        color_img = cv2.imread(path_color+"/%s" % name)
        mask_img = cv2.imread(path_mask+"/%s" % name)

        # Matching both the color and the mask images
        # print(color_img.shape,'\t',mask_img.shape)
        # color_img = cv2.resize(color_img, (mask_img.shape[1], mask_img.shape[0]))
        mask_img = cv2.resize(mask_img, (color_img.shape[1], color_img.shape[0]))
        # print(color_img.shape,'\t',mask_img.shape)

        # Change Format
        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # resize images
        # color_img = cv2.resize(color_img, (width, height))
        # mask_img = cv2.resize(mask_img, (width, height))

        # crop images
        color_img = color_img[0:height, 0:width]
        mask_img = mask_img[0:height, 0:width]

        # print(color_img.shape,'\t',mask_img.shape)

        # Threshold images | After all resizing
        # ret_1,color_img = cv2.threshold(color_img,128,255,cv2.THRESH_BINARY)
        ret_2, mask_img = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)

        # View images
        # cv2.imshow('color_img',color_img)
        # cv2.imshow('mask_img',mask_img)
        # cv2.waitKey(0)

        color_img = np.array(color_img)
        mask_img = np.array(mask_img)

        color_img_list.append(color_img)
        mask_img_list.append(mask_img)

        # break

    # converting to numpy arrays
    color_img_array = np.array(color_img_list)
    mask_img_array = np.array(mask_img_list)

    print('\nNOTE','\tcolor_img_array:',len(color_img_array),'\tmask_img_array:',len(mask_img_array),'\n')

    return color_img_array, mask_img_array

def get_images_multi_croped(path_color=None,path_mask=None,height=0,width=0):
    # path_color = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color'
    # path_true = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_mask'
    # path_color = '/disk1/2015cs120/Dataset/split/on_road_train_color'
    # path_mask = '/disk1/2015cs120/Dataset/split/on_road_train_mask'
    
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
        ret_2, mask_img = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)

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
            # cv2.imshow('color_crop_bottom_left',color_crop_bottom_right)
            # cv2.imshow('mask_img',mask_img)
            # cv2.imshow('mask_crop_top',mask_crop_bottom)
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

        # break

    # converting to numpy arrays
    color_img_array = np.array(color_img_list)
    mask_img_array = np.array(mask_img_list)

    print('@\t','color_img_array:\t',len(color_img_array),'mask_img_array:\t',len(mask_img_array))

    return color_img_array, mask_img_array

def RA_unit_3(x, h, w, n):
    print("\nx, and n :", x.shape, n,'\n')
    x_1 = tf.nn.avg_pool2d(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    print('x_1\t',np.shape(x_1),'')
    print('x_t\t',np.shape(x_t),'\n')
    for k in range(n):
        x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,int(w/2),x.shape[3].value])
        x_t_2 = tf.image.resize(x_t_1, [h,w], 1)
        x_t_3 = tf.abs(x - x_t_2)
        x_t = tf.concat([x_t, x_t_3], axis=3)
        print('x_t_1\t',np.shape(x_t_1))
        print('x_t_2\t',np.shape(x_t_2))
        print('x_t_3\t',np.shape(x_t_3))
        print('x_t\t'  ,np.shape(x_t)  ,'\n')
    x_out = tf.concat([x, x_t], axis=3) # [h,w,c(n+1))] | 3[8+1] = 27

    conv = Conv2D(x.shape[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x_out)
    
    # conv = tf.nn.max_pool2d(conv, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
    conv = MaxPooling2D(pool_size=(1,1),strides=(1,1),padding="same")(conv)
    
    print('\nx_out\t',np.shape(x_out),x_out,'\n')
    print('\nconv\t',np.shape(conv),conv,'\n')
    print('\nx\t',np.shape(x),x,'\n')
    
    # x_p = tf.Session().run(x)
    # print('\nx_p\t',x_p,'\n')

    return x

def RA_unit_2(x, h, w, n):
    print("@ x,h and n :",x.shape,h,n)
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    x_2 = tf.zeros([1, h, w, 0], tf.float32)
    x_out = None
    
    for k in range(n):
        # tf.slice(input,begin,size,name=None)
        x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,int(w/2),x.shape[3].value])
        x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1)
        x_2 = tf.concat([x_2, x_t_2], axis=3)
        x_t_3 = tf.abs(x - x_t_2)
        x_t = tf.concat([x_t, x_t_3], axis=3)
    x_out = tf.concat([x, x_t], axis=3)
    
    # return x_out , x_2
    return x_out

def RA_unit_1(x, h, w, n):
    print("h and n :",h,n)
    # x = tf.reshape(x,(1,x.shape[0],x.shape[1],x.shape[3]))
    # x-input, h-height, w-width, n-slice count
    # if the image height is h and required slice count is n, then the slice height is h/n. So the stride is also h/n.
    # tf.nn.avg_pool(input,ksize,strides,padding,data_format=None,name=Non) one image one channel
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    x_2 = tf.zeros([1, h, w, 0], tf.float32)
    x_out = None
    # x_t_small = tf.zeros([1, x_1.shape[1].value, w/2, 0], tf.float32)
    for k in range(n):
        # tf.slice(input,begin,size,name=None)
        x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,int(w/2),x.shape[3].value])
        x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1)
        x_2 = tf.concat([x_2, x_t_2], axis=3)
        x_t_3 = tf.abs(x - x_t_2)
        x_t = tf.concat([x_t, x_t_3], axis=3)
    x_out = tf.concat([x, x_t], axis=3)
    # return x_out , x_2
    # x_out = tf.reshape(x_out,(x_out.shape[1],x_out.shape[2],x_out.shape[3]))
    # x_out = tf.reshape(x_out)
    # print('x_out',x_out.shape)
    return x_out

def RA_unit_0(x=None, h=0, w=0, n=0):
    # x-input, h-height, w-width, n-slice count
    
    h = x.shape[0].value
    w = x.shape[1].value
    n = n

    print("h, w and n :", h, w, n)

    h = int(h)
    w = int(w)
    n = int(n)
    # z = int(h/n)

    # if the image height is h and required slice count is n, then the slice height is h/n. So the stride is also h/n.
    # https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool
    # tf.nn.avg_pool(input,ksize,strides,padding,data_format=None,name=Non)
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1],strides=[1, h/n, 2, 1], padding="SAME")

    x_t = tf.zeros([1, h, w, 0], tf.float32)

    for k in range(n):
        # c = int(x.shape[3].value)
        x_t_1 = tf.slice(x_1, [0, k, 0, 0], [1, 1, int(w/2), int(x.shape[3].value)]) # tf.slice(input,begin,size,name=None)
        x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1) # resize back to h,w
        x_t_3 = tf.abs(x - x_t_2) # i'-x'
        x_t = tf.concat([x_t, x_t_3], axis=3) # concatenating to obtain D = I' - X' = k(i'-x')
        
        print('x_t_2\n',x_t_2[0],'\n')
        # print('x\n',x[0],'\n')

        x_t_2 = tf.Session().run(x_t_2)
        x = tf.Session().run(x)

        print('x_t_2\n',x_t_2[0],'\n')
        # print('x\n',x[0],'\n')

        x_t_2 = x_t_2.astype(np.uint8) #'int8'
        x = x.astype(np.uint8)

        print(type(x_t_2), x_t_2.dtype, np.shape(x_t_2[0]))
        print(type(x), x.dtype, np.shape(x[0]))

        print('x_t_2\n',x_t_2[0],'\n')
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
    # print('x_out\t',np.shape(x_out),'\n')

    return x_out
    
def unet(pretrained_weights = None,input_size = None):
    inputs = Input(batch_shape=input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = RA_unit_3(x=pool1,h=pool1.shape[1].value, w=pool1.shape[2].value,n=16)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # pool2 = RA_unit_3(x=pool2,h=pool2.shape[1].value, w=pool2.shape[2].value,n=16)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # pool3 = RA_unit_3(x=pool3,h=pool3.shape[1].value, w=pool3.shape[2].value,n=16)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # pool4 = RA_unit_3(x=pool4,h=pool4.shape[1].value, w=pool4.shape[2].value,n=16)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)

    # merge6 = RA_unit(x=merge6,n=16)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)

    # merge7 = RA_unit(x=merge7,n=16)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)

    # merge8 = RA_unit(x=merge8,n=16)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)

    # merge9 = RA_unit(x=merge9,n=16)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy']) # original 1e-4 | 2e-4 = 0.00020
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__=='__main__':
    #Note: tensorflow.python.framework.errors_impl.ResourceExhaustedError:
    #Reduce batch size as @ https://stackoverflow.com/questions/46066850/understanding-the-resourceexhaustederror-oom-when-allocating-tensor-with-shape

    batch_size = 1
    epochs = 50
 
    # Note : FCN8s shape - color (360, 640, 3) & mask (720, 1280, 3) | half (360,640) 
    # image_height = 256
    # image_width = 512
    image_height = 352 #360 
    image_width = 640

    # root_path = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/' 
    root_path = '/home/harin/split/'


    color_img_array_train,mask_img_array_train=get_images_resized(path_color=root_path+'on_road_train_color',path_mask=root_path+'on_road_train_mask',height=image_height,width=image_width)
    color_img_array_test,mask_img_array_test=get_images_resized(path_color=root_path+'on_road_test_color',path_mask=root_path+'on_road_test_mask',height=image_height,width=image_width)
    model=unet(input_size = (1,image_height,image_width,3))
    # model=unet(input_size = (1,image_height,image_width,3)) #(256,256,3) (256,512,3) (256,256,1)
    # model.fit(x=color_img_array_train,y=mask_img_array_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,shuffle=True)
    # model.save(root_path+'unet_model_on_road.h5')
    # model.predict(color_img_array_test)
    print('saved: unet_model_on_road')
    
    # color_img_array,mask_img_array=get_images(path_color=root_path+'off_road_train_color',path_mask=root_path+'off_road_train_mask')
    # model=unet(input_size = (256,256,3))
    # model.fit(x=color_img_array,y=mask_img_array,batch_size=batch_size,epochs=epochs,shuffle=True)
    # model.save(root_path+'unet_model_off_road.h5')
    # print('unet_model_off_road')

    # color_img_array,mask_img_array=get_images(path_color=root_path+'both_road_train_color',path_mask=root_path+'both_road_train_mask')
    # model=unet(input_size = (256,256,3))
    # model.fit(x=color_img_array,y=mask_img_array,batch_size=batch_size,epochs=epochs,shuffle=True)
    # model.save(root_path+'unet_model_both_road.h5')
    # print('unet_model_both_road')