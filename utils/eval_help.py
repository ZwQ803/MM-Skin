import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score, brier_score_loss, log_loss, classification_report)
import netcal.metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score




def binary_metrics(targets, preds, label_set=[0, 1], suffix='', return_arrays=False):
    if len(targets) == 0:
        return {}
    # print("targets:", targets, "preds:", preds)

    res = {
        'accuracy': accuracy_score(targets, preds),
        'n_samples': len(targets)
    }

    # if len(label_set) == 2:
    # 不进行下面的操作if判断条件不存在
    if len(label_set) == 100:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item()
        res['FN'] = CM[1][0].item()
        res['TP'] = CM[1][1].item()
        res['FP'] = CM[0][1].item()

        res['error'] = res['FN'] + res['FP']

        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP']/(res['TP']+res['FN'])
            res['FNR'] = res['FN']/(res['TP']+res['FN'])

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP']/(res['FP']+res['TN'])
            res['TNR'] = res['TN']/(res['FP']+res['TN'])

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
    else:
        CM = confusion_matrix(targets, preds, labels=label_set)
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    # 计算宏观平均
    res['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0.)
    res['precision_macro'] = precision_score(targets, preds, average='macro', zero_division=0.)
    res['recall_macro'] = recall_score(targets, preds, average='macro', zero_division=0.)


    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds
    print("binary metrics finished")
    return {f"{i}{suffix}": res[i] for i in res}


def prob_metrics(targets, preds, label_set, return_arrays=False):
    # print("targets:", targets, "preds:", preds)
    # label = np.unique(targets)
    if len(targets) == 0:
        return {}
    # 获取真实标签和预测值中共同的标签
    # unique_labels = np.unique(targets)
    # preds_labels = np.unique(np.argmax(preds, axis=1))  # 获取预测结果中实际出现的标签
    # print("label_set:", label_set)
    if preds.ndim == 1 or preds.shape[1] == 1:  # 二分类
        preds_labels = (preds >= 0.5).astype(int)  # 预测类为1的概率大于0.5时，标记为1，否则为0
    else:  # 多分类
        preds_labels = np.argmax(preds, axis=1)  # 获取每个样本的最大概率对应的标签

    res = {
        'BCE': log_loss(targets, preds, labels=label_set),
        'ECE': netcal.metrics.ECE().measure(preds, targets)
    }
    # print("label_set:", label_set)
    unique_labels = np.unique(targets)
    # print("unique_labels:", unique_labels)

    # if len(set(targets)) > 2:
    #     # happens when you predict a class, but there are no samples with that class in the dataset
    #     try:
    #         res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=unique_labels)
    #     except:
    #         res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set)
    # elif len(set(targets)) == 2:
    #     res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr',labels=label_set)
    # elif len(set(targets)) == 1:
    #     res['AUROC'] = None
    res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set)
    res['AUC_avg'] = roc_auc_score(targets, preds, average='macro', multi_class='ovr', labels=label_set)
    res['AUPRC'] = average_precision_score(targets, preds,average='macro')

    # if len(set(targets)) == 2:
    #     # res['ROC_curve'] = roc_curve(targets, preds)
    #     res['AUPRC'] = average_precision_score(targets, preds, average='macro')
    #     res['brier'] = brier_score_loss(targets, preds)
    #     res['mean_pred_1'] = preds[targets == 1].mean()
    #     res['mean_pred_0'] = preds[targets == 0].mean()

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res