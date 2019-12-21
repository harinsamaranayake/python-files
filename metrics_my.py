# note : navigate to the 'test' folder which have the two folders as 'truth' and 'pred'
import os
import cv2


def metrics():
    path_true = "/Users/harinsamaranayake/Desktop/test/truth"
    path_pred = "/Users/harinsamaranayake/Desktop/test/pred"
    true_list = next(os.walk(path_true))[2]
    pred_list = next(os.walk(path_pred))[2]
    true_list.remove('.DS_Store')
    pred_list.remove('.DS_Store')

    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []

    for file_name in true_list:
        true_img = cv2.imread(
            "/Users/harinsamaranayake/Desktop/test/truth/%s" % file_name)
        pred_img = cv2.imread(
            "/Users/harinsamaranayake/Desktop/test/pred/%s" % file_name)

        print(true_img.shape)
        print(pred_img.shape)
        height = true_img.shape[0]
        width = true_img.shape[1]

        # water
        true_positive = 0
        # ground
        true_negative = 0
        false_positive = 0
        false_negative = 0
        # intersections
        intersect_white = 0
        intersect_black = 0
        # unions
        union_white = 0
        union_black = 0

        if True:
            for x in range(width):
                for y in range(height):
                    if((pred_img[y][x] == true_img[y][x]) == 1):
                        true_positive += 1
                        intersect_white += 1
                    elif((pred_img[y][x] == true_img[y][x]) == 0):
                        true_negative += 1
                        intersect_black += 1
                    elif(pred_img[y][x] != true_img[y][x]):
                        if(pred_img[y][x] == 1):
                            false_positive += 1
                        elif(pred_img[y][x] == 0):
                            false_negative += 1

                    if(pred_img[y][x] == 1):
                        union_white += 1
                    elif(pred_img[y][x] == 0):
                        union_black += 1
                    if(true_img[y][x] == 1):
                        union_white += 1
                    elif(true_img[y][x] == 0):
                        union_black += 1

            precision = true_positive/(true_positive+false_positive)
            recall = true_positive/(true_positive+false_negative)
            f1_score = 2*(precision*recall)/(precision+recall)
            iou = (intersect_white/union_white)+(intersect_black/union_black)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)
            iou_list.append(iou)

    return precision_list, recall_list, f1_list, iou_list


if __name__ == "__main__":
    precision_list, recall_list, f1_list, iou_list = metrics()
    print(precision_list)
