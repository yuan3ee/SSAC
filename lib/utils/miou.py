from sklearn.metrics import confusion_matrix
import numpy as np


def get_metrics(pre, label, num, ignore=-1):
    pre = pre.reshape(-1)
    label = label.reshape(-1)
    ignore_index = pre != ignore
    seg_pre = pre[ignore_index]
    seg_label = label[ignore_index]
    label_value = list()
    for i in range(num):
        label_value.append(i)
    confusion_matrix_ = confusion_matrix(seg_pre, seg_label, labels=label_value)
    return confusion_matrix_

def miou(confusion_matrix):
    confusion_temp = np.zeros((2,2))
    iou = list()
    nums = confusion_matrix.shape[0]
    iou_all = 0
    for i in range(nums):
        TP = confusion_matrix[i][i]
        temp = np.concatenate((confusion_matrix[0:i, :], confusion_matrix[i+1:,:]), axis=0)
        sum_one = np.sum(temp, axis=0)
        FP = sum_one[i]

        temp2 = np.concatenate((confusion_matrix[:, 0:i], confusion_matrix[:, i + 1:]), axis=1)
        FN = np.sum(temp2, axis=1)[i]
        TN = temp2.reshape(-1).sum() - FN
        iou_temp = TP/(TP+FP+FN)
        iou_all += iou_temp
        iou.append(iou_temp)
    mean_iou = iou_all/nums
    return mean_iou, iou

