import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#### CHANGE
flag_resize_pred=True
path_pred = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_pred'
path_true = '/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_mask'
#### END

true_img_list = []
pred_img_list = []

true_list_new=[]
pred_list_new=[]

pred_img_length = 0
pred_img_width = 0
true_img_length = 0
true_img_width = 0

#### Read from file
def read_from_file():
    text_file_path='/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_mask'
    p=np.genfromtxt(text_file_path,dtype='str')

    for i in range(p.shape[0]):
        img = p[i,0]

        # to obtain the image sizes
        pred_img = cv2.imread(path_pred+"/%s" % img)
        true_img = cv2.imread(path_true+"/%s" % img)

        # print('pred_img',pred_img.shape)
        # print('true_img',true_img.shape)

        pred_img_length = pred_img.shape[0]
        pred_img_width = pred_img.shape[1]
        true_img_length = true_img.shape[0]
        true_img_width = true_img.shape[1]
        # print(pred_img_length, pred_img_width, true_img_length, true_img_width)
        
        break

    for i in range(p.shape[0]):
        # gt imgage name
        img = p[i,0]

        pred_img = cv2.imread(path_pred+"/%s" % img)
        true_img = cv2.imread(path_true+"/%s" % img)

        if (flag_resize_pred):
            pred_img = cv2.resize(pred_img, (true_img_width, true_img_length))
        else:
            true_img = cv2.resize(true_img, (pred_img_width, pred_img_length))

        # Threshold images
        ret_1,pred_img = cv2.threshold(pred_img,128,255,cv2.THRESH_BINARY)
        ret_2,true_img = cv2.threshold(true_img,128,255,cv2.THRESH_BINARY)

        true_img_list.append(true_img)
        pred_img_list.append(pred_img)

    #converting to numpy arrays
    true_list_new = np.array(true_img_list)
    pred_list_new = np.array(pred_img_list)

    #flatterning the arrays
    true_list_new = true_list_new.flatten()
    pred_list_new = pred_list_new.flatten()

#### Read from folder
# true_list = next(os.walk(path_true))[2]
pred_list = next(os.walk(path_pred))[2]

if '.DS_Store' in pred_list:
    pred_list.remove('.DS_Store')

for img in pred_list:
    # to obtain the image sizes
    pred_img = cv2.imread(path_pred+"/%s" % img)
    true_img = cv2.imread(path_true+"/%s" % img)

    print('pred_img',pred_img.shape)
    print('true_img',true_img.shape)

    pred_img_length = pred_img.shape[0]
    pred_img_width = pred_img.shape[1]
    true_img_length = true_img.shape[0]
    true_img_width = true_img.shape[1]
    # print(pred_img_length, pred_img_width, true_img_length, true_img_width)
    
    break

for img in pred_list:
    #to obtain all predicted and gt images
    pred_img = cv2.imread(path_pred+"/%s" % img)
    true_img = cv2.imread(path_true+"/%s" % img)

    # Making the both images same size
    if (flag_resize_pred):
        pred_img = cv2.resize(pred_img, (true_img_width, true_img_length))
    else:
        true_img = cv2.resize(true_img, (pred_img_width, pred_img_length))

    # Threshold images
    ret_1,pred_img = cv2.threshold(pred_img,128,255,cv2.THRESH_BINARY)
    ret_2,true_img = cv2.threshold(true_img,128,255,cv2.THRESH_BINARY)

    # View images
    # cv2.imshow('p',pred_img)
    # cv2.imshow('t',true_img)
    # cv2.waitKey(0)

    # pred_img=np.array(pred_img)
    # true_img=np.array(true_img)
    # pred_img=pred_img.flatten()
    # true_img=true_img.flatten()

    true_img_list.append(true_img)
    pred_img_list.append(pred_img)

#converting to numpy arrays
true_list_new = np.array(true_img_list)
pred_list_new = np.array(pred_img_list)

#flatterning the arrays
true_list_new = true_list_new.flatten()
pred_list_new = pred_list_new.flatten()

def confusion_metrics_method_01():
    print('START > confusion_metrics_method_01')

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(0, len(true_list_new)):
        true_value = true_list_new[i]
        pred_value = pred_list_new[i]

        if((pred_value == 255) and (pred_value == true_value)):
            tp += 1
        elif((pred_value == 255) and (pred_value != true_value)):
            fp += 1
        elif((pred_value == 0) and (pred_value != true_value)):
            fn += 1
        elif ((pred_value == 0) and (pred_value == true_value)):
            tn += 1

        if ((i % 100000) == 0):
            print(i)

    print('tp', tp, '\tfp', fp, '\tfn', fn, '\ttn', tn)

    try:
        # water : white
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        union_water=tp
        intersection_water=fn+fp
        iou_water=union_water/intersection_water
        print('precision_water', precision, '\t recall_water', recall,'\t iou_water',iou_water)

        # background : black
        precision = tn/(tn+fn)
        recall = tn/(tn+fp)
        print('precision', precision, '\t', 'recall', recall)

        union_ground=tn
        intersection_ground=fn+fp
        iou_ground=union_ground/intersection_ground
        print('precision_ground', precision, '\t recall_ground', recall,'\t iou_ground',iou_ground)
   
    except:
        print('calculation error')

def confusion_metrics_method_02():
    pass

def list_value_finder():
    # List value finder. Get different values that exists within a list.
    print('START > List value finder')

    true_value_list = []
    pred_value_list = []

    for i in true_list_new:
        if i in true_value_list:
            pass
        else:
            true_value_list.append(i)

    for j in pred_list_new:
        if j in pred_value_list:
            pass
        else:
            pred_value_list.append(j)

    print(true_value_list)
    print(pred_value_list)

    print(len(true_list_new))
    print(len(pred_list_new))

def scikit_metrix():
    y_true = true_list_new
    y_pred = pred_list_new

    cf = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix: \n", cf)

    p1 = cf[0, 0]/(cf[0, 0]+cf[0, 1])
    r1 = cf[0, 0]/(cf[0, 0]+cf[1, 0])
    p2 = cf[1, 1]/(cf[1, 1]+cf[1, 0])
    r2 = cf[1, 1]/(cf[1, 1]+cf[0, 1])
    print('precision-water',p1, '\t recall-water', r1, '\t precision-ground', p2, '\t recall-ground', r2)

    # print("Accuracy : ", accuracy_score(y_true=y_true, y_pred=y_pred)*100)
    # print("Precision : ", precision_score(y_true=y_true, y_pred=y_pred, pos_label=0)*100)
    # print("Precision : ", precision_score(y_true=y_true, y_pred=y_pred, pos_label=255)*100)
    print("Precision : ", precision_score(y_true=y_true, y_pred=y_pred,average='weighted')*100)
    # print("Recall : ", recall_score(y_true=y_true, y_pred=y_pred, average='weighted')*100)
    # print("F1_Score : ", f1_score(y_true=y_true, y_pred=y_pred, average='weighted')*100)
    print('Classification Report: \n', classification_report(y_true=y_true, y_pred=y_pred))

# def computeIoU(y_pred_batch, y_true_batch):
#     return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]))

# def pixelAccuracy(y_pred, y_true):
#     y_pred = np.argmax(np.reshape(y_pred,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
#     y_true = np.argmax(np.reshape(y_true,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
#     y_pred = y_pred * (y_true>0)
#     return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)

if __name__ == '__main__':
    # read_from_folder()
    # confusion_metrics_method_01()
    # list_value_finder()
    # scikit_metrix()
    pass
