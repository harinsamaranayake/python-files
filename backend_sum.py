import cv2
from sklearn import metrics
from keras import backend as K
smooth = 1.

y_true=cv2.imread("img_0000000101_gt.png")
y_true=cv2.resize(y_true,(640,360))
y_true=K.flatten(y_true)
y_pred=cv2.imread("img_000000101.png")
y_pred=K.flatten(y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[0,1,2])
    print(intersection)
    # union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    # return K.mean( (2 * intersection + smooth) / (union + smooth), axis=0)

def bce_dice(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)-K.log(dice_coef(y_true, y_pred))

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

# seg_model.compile(optimizer = 'adam',
#               loss = bce_dice, 
#               metrics = ['binary_accuracy', dice_coef, true_positive_rate])

if __name__ == "__main__":
    # metrics.precision_score(K.flatten(y_pred),K.flatten(y_true))
    # print(dice_coef(y_true, y_pred))
    print(y_pred)
    print(y_true)
    pass