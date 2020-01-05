# Link - https://github.com/zhixuhao/unet/blob/master/model.py
# python UNet_02V2.py &> out.txt
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

def get_images(path_color,path_mask):
    # path_color = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color'
    # path_true = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_mask'
    # path_color = '/disk1/2015cs120/Dataset/split/on_road_train_color'
    # path_mask = '/disk1/2015cs120/Dataset/split/on_road_train_mask'
    
    #File name at location 02
    # try:
    #     color_name_list = next(os.walk(path_color))[2]
    # except StopIteration:
    #     pass
    print(path_color)
    color_name_list = next(os.walk(path_color))[2]

    if '.DS_Store' in color_name_list:
        color_name_list.remove('.DS_Store')

    color_img_list = []
    mask_img_list = []

    for name in color_name_list:
        #to obtain predicted and gt images
        color_img = cv2.imread(path_color+"/%s" % name)
        mask_img = cv2.imread(path_mask+"/%s" % name)

        #Check shape and match both sizes
        # print(color_img.shape)
        # print(mask_img.shape)
        # color_img = cv2.resize(color_img, (mask_img.shape[1], mask_img.shape[0]))
        # mask_img = cv2.resize(mask_img, (color_img.shape[1], color_img.shape[0]))
        # print(color_img.shape)
        # print(mask_img.shape)

        # Change Format
        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # Resize Images
        color_img = cv2.resize(color_img, (image_width, image_height))
        mask_img = cv2.resize(mask_img, (image_width, image_height))
        print(color_img.shape)
        print(mask_img.shape)

        # Threshold images
        # ret_1,color_img = cv2.threshold(color_img,128,255,cv2.THRESH_BINARY)
        ret_2,mask_img = cv2.threshold(mask_img,128,255,cv2.THRESH_BINARY)

        # View images
        # cv2.imshow('color_img',color_img)
        # cv2.imshow('mask_img',mask_img)
        # cv2.waitKey(0)

        color_img=np.array(color_img)
        mask_img=np.array(mask_img)

        # color_img.reshape(256,256,1)
        # print(color_img.shape)

        color_img_list.append(color_img)
        mask_img_list.append(mask_img)
        
        # break

    # converting to numpy arrays
    color_img_array = np.array(color_img_list)
    mask_img_array = np.array(mask_img_list)

    return color_img_array,mask_img_array
    
def unet(pretrained_weights = None,input_size = (256,256,1)):
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
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)#conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__=='__main__':
    #Note: tensorflow.python.framework.errors_impl.ResourceExhaustedError:
    #Reduce batch size as @ https://stackoverflow.com/questions/46066850/understanding-the-resourceexhaustederror-oom-when-allocating-tensor-with-shape

    batch_size = 1
    epochs = 2
 
    image_size=256
    image_height = 256
    image_width = 512

    root_path = '/home/harin/split/'

    color_img_array,mask_img_array=get_images(path_color=root_path+'on_road_train_color',path_mask=root_path+'on_road_train_mask')
    model=unet(input_size = (image_height,image_width,3)) #(256,256,3) (256,512,3)
    model.fit(x=color_img_array,y=mask_img_array,batch_size=batch_size,epochs=epochs,validation_split=0.2,shuffle=True)
    model.save(root_path+'unet_model_on_road.h5')
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