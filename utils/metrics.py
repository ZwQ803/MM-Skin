"""
下游任务迁移的评价指标
管理评价指标，k折交叉验证，结果保存
Evaluation metric of downstream task migration
Management evaluation metric, K-fold cross-validation, results saving
"""

import os
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, recall_score
import csv




# 主控函数调用接口  main function call interface
def evaluate(refs, preds, task="classification"):
    if task == "classification":
        metrics = classification_metrics(refs, preds)
        print(
            'Metrics: aca=%2.5f - TAOP_aca=%2.5f - kappa=%2.3f - macro f1=%2.3f - auc=%2.3f '
            % (metrics["aca"], metrics["TAOP_acc"], metrics["kappa"], metrics["f1_avg"], metrics["auc_avg"]))
    elif task == "segmentation":
        metrics = segmentation_metrics(refs, preds)
        print('Metrics: dsc=%2.5f - auprc=%2.3f' % (metrics["dsc"], metrics["auprc"]))
    else:
        metrics = {}
    return metrics


# auc
def au_prc(true_mask, pred_mask):
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[np.isnan(f1)] = 0
    th = threshold[np.argmax(f1)]

    return au_prc, th


# 特异性计算  specificity
def specificity(refs, preds):
    TN = np.sum((refs == 0) & (preds == 0))
    FP = np.sum((refs == 0) & (preds == 1))
    if TN + FP == 0:
        return 0.0
    return TN / (TN + FP)


# 分类评价指标  Classification evaluation metric
def classification_metrics(refs, preds):
    k = np.round(cohen_kappa_score(refs, np.argmax(preds, -1), weights="quadratic"), 3)  # Kappa quadatic

    cm = confusion_matrix(refs, np.argmax(preds, -1))  # 混淆矩阵  confusion matrix
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.nanmean(np.diag(cm_norm)), 4) # 使用 np.nanmean 忽略 nan 值

    recall_class = [np.round(recall_score(refs == i, np.argmax(preds, -1) == i), 3) for i in
                    np.unique(refs)]  # 每个类别的召回率  recall per class
    recall_avg = np.round(np.nanmean(recall_class), 4)  # 平均召回率  average recall

    specificity_class = [np.round(specificity(refs == i, np.argmax(preds, -1) == i), 3) for i in
                         np.unique(refs)]  # 每个类的特异性 specificity
    specificity_avg = np.round(np.nanmean(specificity_class), 4)  # 平均特异性  average specificity

    auc_class = [np.round(roc_auc_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]  # 每个类的auc auc
    auc_avg = np.round(np.nanmean(auc_class), 4)  # 平均auc  average auc

    f1_class = [np.round(f1_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]  # 每个类的f1  f1
    f1_avg = np.round(np.nanmean(f1_class), 4)  # 平均f1  average f1

    # AMD dataset
    # AMD_auc = roc_auc_score(refs, preds[:,1])
    # precision, recall, thresholds = precision_recall_curve(refs, preds[:,1])
    # kappa_list = []
    # f1_list = []
    # for threshold in thresholds:
    #     y_scores = preds[:,1]
    #     y_scores = np.array(y_scores >= threshold, dtype=float)
    #     kappa = cohen_kappa_score(refs, y_scores)
    #     kappa_list.append(kappa)
    #     f1 = f1_score(refs, y_scores)
    #     f1_list.append(f1)
    # kappa_f1 = np.array(kappa_list) + np.array(f1_list)
    # AMD_kappa = kappa_list[np.argmax(kappa_f1)]
    # AMD_f1 = f1_list[np.argmax(kappa_f1)]

    # TAOP dataset
    class_acc = accuracy_score(refs, np.argmax(preds, -1))

    metrics = {"aca": aca, "TAOP_acc": class_acc, "acc_class": acc_class,
               "kappa": k,
               "auc_class": auc_class, "auc_avg": auc_avg,
               "f1_class": f1_class, "f1_avg": f1_avg,
               "sensitivity_class": recall_class, "sensitivity_avg": recall_avg,
               "specificity_class": specificity_class, "specificity_avg": specificity_avg,
               "cm": cm, "cm_norm": cm_norm}
    return metrics


# K折交叉验证平均  K-fold cross-verify average
def average_folds_results(list_folds_results, task):
    '''
    list_folds_results：存放了K折交叉验证的结果
    '''
    metrics_name = list(list_folds_results[0].keys())  # 全部评价指标的名称  Name of all evaluation metric

    out = {}
    for iMetric in metrics_name:
        if iMetric in ['cm', 'cm_norm']:
            # 平均混淆矩阵
            cm_sum = np.sum([fold_result[iMetric] for fold_result in list_folds_results], axis=0)
            if iMetric == 'cm':
                out[iMetric + "_avg"] = cm_sum.tolist()
            elif iMetric == 'cm_norm':
                # 归一化混淆矩阵
                cm_norm_avg = cm_sum / cm_sum.sum(axis=1, keepdims=True)
                out[iMetric + "_avg"] = cm_norm_avg.tolist()
        else:
            values = np.concatenate([np.expand_dims(np.array(iFold[iMetric]), -1) for iFold in list_folds_results], -1)
            out[(iMetric + "_avg")] = np.round(np.mean(values, -1), 3).tolist()
            out[(iMetric + "_std")] = np.round(np.std(values, -1), 3).tolist()

    if task == "classification":
        print('Metrics: aca=%2.3f(%2.3f) - TAOP_aca=%2.3f(%2.3f) - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f) -' % (
                  out["aca_avg"], out["aca_std"], out["TAOP_acc_avg"], out["TAOP_acc_std"],
                  out["kappa_avg"], out["kappa_std"], out["f1_avg_avg"], out["f1_avg_std"]))

    return out


def save_heatmap(val_cm, cls, heatmap_result_dir):
    xtick = cls.keys()
    ytick = cls.keys()
    # 处理分母为零的情况，将分母为零的行的所有元素设置为0，以避免除法错误
    plt.clf()
    # val_cm = np.where(val_cm.sum(axis=1)[:, np.newaxis] == 0, 0, val_cm)
    val_cm = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]
    h = sns.heatmap(val_cm, fmt='.2f', cmap="Blues", annot=True, cbar=False, xticklabels=xtick, yticklabels=ytick)
    cb = h.figure.colorbar(h.collections[0])
    plt.savefig(os.path.join(heatmap_result_dir, 'confusionmatrix.pdf'))


# 保存实验结果 适配器权重  Save experiment results and adapter weights
def save_results(metrics, out_path, id_experiment=None, id_metrics=None, save_model=False, weights=None, class_labels=None):
    '''
    metrics：K折交叉验证的结果  Results of K-fold cross-validation
    id_experiment：实验id
    '''
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if id_experiment is None:
        id_experiment = "experiment" + str(np.random.rand())
    else:
        id_experiment = id_experiment
    if not os.path.isdir(out_path + id_experiment):
        os.mkdir(out_path + id_experiment)

    experiment_dir = out_path + id_experiment
    # 以json格式保存结果  Save the results in json format
    with open(out_path + id_experiment + '/metrics_' + id_metrics + '.json', 'w') as fp:
        json.dump(metrics, fp)

    # 保存实验结果为csv文件  Save the experiment results as a csv file，以csv格式保存结果
    with open(out_path + id_experiment + '/metrics_' + id_metrics + '.csv', 'w') as f:
        writer = csv.writer(f)
        headers = list(metrics.keys())
        writer.writerow(headers)
        row = []
        for key in headers:
            value = metrics[key]
            if isinstance(value, list):
                value = json.dumps(value)  # 将列表转换为JSON字符串
            row.append(value)
        writer.writerow(row)

    print(f"Metrics successfully saved to")


    # csv格式保存结果  Save the results in csv format
    # metrics_df = pd.DataFrame(metrics)
    # detailed_metrics = {}
    # for key, value in metrics.items():
    #     if isinstance(value, list):
    #         detailed_metrics[key] = value
    #     else:
    #         detailed_metrics[key] = [value]
    # detailed_metrics_df = pd.DataFrame(detailed_metrics)
    # detailed_metrics_csv_path = os.path.join(experiment_dir, 'detailed_metrics.csv')
    # detailed_metrics_df.to_csv(detailed_metrics_csv_path, index=False)
    # print(f"详细评估指标已保存到 CSV 文件: {detailed_metrics_csv_path}")

    # 保存混淆矩阵  Save confusion matrix
    if 'cm_avg' in metrics and class_labels is not None:
        cm_avg = np.array(metrics['cm_avg'])
        cm_df = pd.DataFrame(cm_avg, index=class_labels, columns=class_labels)
        cm_csv_path = os.path.join(experiment_dir, 'confusion_matrix_avg.csv')
        cm_df.to_csv(cm_csv_path)
        print(f"平均混淆矩阵已保存到 CSV 文件: {cm_csv_path}")

        # 绘制并保存混淆矩阵热力图
        heatmap_result_dir = experiment_dir  # 可以根据需要更改为子目录
        save_heatmap(cm_avg, class_labels, heatmap_result_dir)

    # 保存实验结果为csv文件  Save the experiment results as a csv file
    # 移除混淆矩阵相关的键
    # metrics.pop('cm', None)
    # metrics.pop('cm_norm', None)
    # metrics.pop('cm_avg', None)
    # metrics.pop('cm_norm_avg', None)
    # metrics_df = pd.DataFrame(metrics)
    # metrics_csv_path = os.path.join(experiment_dir, 'metrics.csv')
    # metrics_df.to_csv(metrics_csv_path, index=False)


    # 保存适配器权重  Save adapter weight
    if save_model:
        import torch
        for i in range(len(weights)):
            torch.save(weights[i], out_path + id_experiment + '/weights_' + str(i) + '.pth')
