# Link - https://github.com/zhixuhao/unet/blob/master/model.py
from __future__ import absolute_import, division, print_function, unicode_literals
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
import tensorflow as tf
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
    
def unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9) #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy']) # original 1e-4 | 2e-4 = 0.00020
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__=='__main__':
    #Note: tensorflow.python.framework.errors_impl.ResourceExhaustedError:
    #Reduce batch size as @ https://stackoverflow.com/questions/46066850/understanding-the-resourceexhaustederror-oom-when-allocating-tensor-with-shape

    batch_size = 10
    epochs = 400
 
    # Note : FCN8s shape - color (360, 640, 3) & mask (720, 1280, 3) | half (360,640) 
    # image_height = 256
    # image_width = 512
    image_height = 352 #360 
    image_width = 640
 
    root_path = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/' #'/home/harin/split/'

    color_img_array_train,mask_img_array_train=get_images_resized(path_color=root_path+'on_road_train_color',path_mask=root_path+'on_road_train_mask',height=image_height,width=image_width)
    color_img_array_test,mask_img_array_test=get_images_resized(path_color=root_path+'on_road_test_color',path_mask=root_path+'on_road_test_mask',height=image_height,width=image_width)
    model=unet(input_size = (image_height,image_width,3)) #(256,256,3) (256,512,3) (256,256,1)
    # model.fit(x=color_img_array_train,y=mask_img_array_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,shuffle=True)
    # model.save(root_path+'unet_model_on_road.h5')
    # print('saved: unet_model_on_road')
    
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