import numpy as np
from sklearn import metrics


def cal_cm(y_true,y_pred):
    """
    计算混淆矩阵
    """
    y_true=y_true.reshape(1,-1).squeeze()
    y_pred=y_pred.reshape(1,-1).squeeze()
    cm=metrics.confusion_matrix(y_true,y_pred)
    return cm


def intersection_over_union(confusion_matrix):
    """
    计算IoU。输入是一个混淆矩阵。
    """
    intersection = np.diag(confusion_matrix)#交集
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)#并集
    IoU = intersection / union #交并比，即IoU
    return IoU

# 一个案例
y_pred = np.random.randint(0, 3, (32,64)).reshape(1,-1).squeeze()
y_true = np.random.randint(0, 3, (32,64)).reshape(1,-1).squeeze()
cm = cal_cm(y_true, y_pred)
iou = intersection_over_union(cm)
print(iou)
