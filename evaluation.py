import os
import warnings
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# 设置中文支持并增大文字大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 模型评估
def model_evaluation(y_true,y_pred,y_prod,name,):
    def calculate_TP(y, y_pred):
        tp = 0
        for i, j in zip(y, y_pred):
            if i == j == 1:
                tp += 1
        return tp
    def calculate_TN(y, y_pred):
        tn = 0
        for i, j in zip(y, y_pred):
            if i == j == 0:
                tn += 1
        return tn
    def calculate_FP(y, y_pred):
        fp = 0
        for i, j in zip(y, y_pred):
            if i == 0 and j == 1:
                fp += 1
        return fp
    def calculate_FN(y, y_pred):
        fn = 0
        for i, j in zip(y, y_pred):
            if i == 1 and j == 0:
                fn += 1
        return fn
    # TNR = TN / (FP + TN) TNR即为特异度（specificity） spe = TN/float(TN+FP)
    def TNR(y, y_pred):
        tn = calculate_TN(y, y_pred)
        fp = calculate_FP(y, y_pred)
        return tn / (tn + fp)
    # TPR =TP/ (TP+ FN)  TPR即为敏感度（sensitivity） sen = TP/float(TP+FN)
    def TPR(y, y_pred):
        tp = calculate_TP(y, y_pred)
        fn = calculate_FN(y, y_pred)
        return tp / (tp + fn)
    def PPV(y, y_pred):
        tp = calculate_TP(y, y_pred)
        fp = calculate_FP(y, y_pred)
        return tp / (tp + fp)
    def NPV(y, y_pred):
        tn = calculate_TN(y, y_pred)
        fn = calculate_FN(y, y_pred)
        return tn / (tn + fn)

    result = pd.DataFrame({
        'Model': [name],
        'ACC': [round(accuracy_score(y_true, y_pred), 3)],
        'Pre': [round(precision_score(y_true, y_pred), 3)],
        'Recall': [round(recall_score(y_true, y_pred), 3)],
        'F1': [round(f1_score(y_true, y_pred), 3)],
        'AUC': [round(roc_auc_score(y_true, y_prod[:, 1], multi_class='ovo'), 3)],
        'Sp': [round(TNR(y_true, y_pred), 3)],
        'Sn': [round(TPR(y_true, y_pred), 3)],
        'MCC': [round(matthews_corrcoef(y_true, y_pred), 3)],
        'PPV': [round(PPV(y_true, y_pred), 3)],
        'NPV': [round(NPV(y_true, y_pred), 3)],
        'kappa': [round(cohen_kappa_score(y_true, y_pred), 3)]})
    return result
# ROC函数
def ROC(names,results_proba,y_test,colors,save_name):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()
    for i,proba,m in zip(colors,results_proba,names):
        fpr, tpr, _ = roc_curve(y_test, proba[:,1])
        plt.plot(fpr, tpr, color=i, lw=2,label='ROC '+m+'={0:.3f}'.format(auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.legend(loc="lower right")
    plt.title('ROC曲线')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig(os.getcwd() + '/result/' + save_name + '/' + save_name + '_ROC_AUC.png')
    plt.close()
# PRC
def PRC(names, results_proba,y_test, colors, save_name):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()
    for i,k,m in zip(colors,results_proba,names):
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, k[:,1])
        plt.plot(lr_recall, lr_precision, color=i, label= m+' (area = %0.2f)' % average_precision_score(y_test, k[:,1]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(os.getcwd()+'/result/' + save_name + '/' + save_name + '_PRC.png')
    plt.close()
