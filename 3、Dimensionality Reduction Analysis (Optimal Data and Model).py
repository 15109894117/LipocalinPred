import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.utils.np_utils import to_categorical
from evaluation import model_evaluation
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
# 设置中文支持并增大文字大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 构建模型
def Deep_learning_model(model_name):
    if model_name == 'LSTM':
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))  # 添加Dropout层，以防止过拟合
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    elif model_name == 'RNN':
        model = Sequential()
        model.add(SimpleRNN(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))  # 添加Dropout层，以防止过拟合
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    elif model_name == 'CNN':
        model = Sequential()
        model.add(Conv1D(128, activation='relu', kernel_size=1, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Flatten())
        model.add(Dropout(0.2))  # 添加Dropout层，以防止过拟合
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    elif model_name == 'DNN':
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[-1:])))
        model.add(Dropout(0.2))  # 添加Dropout层，以防止过拟合
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    elif model_name == 'CNN_LSTM':
        model = Sequential()
        model.add(
            Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    elif model_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50)
    elif model_name == 'SVM':
        from sklearn.svm import SVC
        model = SVC(probability=True,kernel='linear',C=0.1,tol=0.01)
    elif model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=2, leaf_size=20, p=10)
    return model
# 评估模型
def model_evaluation(y_true,y_pred,y_prod,name):
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
# ROC曲线
def ROC(names, results_proba, train_test, colors, save_name):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()
    for i, k, m,y_test in zip(colors, results_proba, names,train_test):
        fpr, tpr, _ = roc_curve(y_test, k[:, 1])
        plt.plot(fpr, tpr, color=i, lw=2, label= m + '(AUC = {0:.3f}'.format(auc(fpr, tpr))+')')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.legend(loc="lower right")
    plt.title('ROC Curve')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.getcwd() + '/result/' + save_name + '/' + save_name + '_ROC_AUC.png')
    plt.show()
    return plt
# PRC曲线
def PRC(names, results_proba, train_test, colors, save_name):
    import os
    os.makedirs('./result', exist_ok=True)  # 创建result文件夹
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()
    for i,k,m,y_test in zip(colors,results_proba,names,train_test):
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, k[:,1])
        plt.plot(lr_recall, lr_precision, color=i, label= m+' (area = %0.2f)' % average_precision_score(y_test, k[:,1]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + save_name + '/' + save_name + '_PRC.png')
    plt.show()
    return plt
# 模型对比ROC
def ROC_AUC(names,models,colors,save_name):
    import os
    os.makedirs('./result', exist_ok=True)  # 创建result文件夹
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()
    for i,j,m in zip(colors,models,names):
        if m in ['RF','SVM','KNN']:
            fpr, tpr, _ = roc_curve(y_test, j.predict_proba(x_test)[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_test, j.predict(X_test)[:, 1])
        plt.plot(fpr, tpr, color=i, lw=2,label='AUC '+m+'={0:.3f}'.format(auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.legend(loc="lower right")
    plt.title('AUC')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig(os.getcwd() + '/result/' + save_name + '/模型对比_ROC_AUC.png')
    plt.close()
# 模型对比PRC
def PRC_curve(names,models,colors,save_name):
    import os
    os.makedirs('./result', exist_ok=True)  # 创建result文件夹
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()
    for i,j,m in zip(colors,models,names):
        if m in ['RF','SVM','KNN']:
            lr_precision, lr_recall, _ = precision_recall_curve(y_test, j.predict_proba(x_test)[:, 1])
            plt.plot(lr_recall, lr_precision, color=i, label= m+' (area = %0.2f)' % average_precision_score(y_test, j.predict_proba(x_test)[:, 1]))
        else:
            lr_precision, lr_recall, _ = precision_recall_curve(y_test, j.predict(X_test)[:, 1])
            plt.plot(lr_recall, lr_precision, color=i, label= m+' (area = %0.2f)' % average_precision_score(y_test, j.predict(X_test)[:, 1]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(os.getcwd()+'/result/'+save_name+'/模型对比_PRC.png')
    plt.close()
# loss损失值
def loss(history,name):
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train-loss')
    plt.plot(history.history['val_loss'], label='test-loss')
    plt.legend()
    plt.title('loss曲线')
    plt.savefig(os.getcwd() + '/result/' + name + '/' + name + '_loss.png')
    plt.show()
    plt.close()
# acc准确率
def acc(history,name):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train-accuracy')
    plt.plot(history.history['val_accuracy'], label='test-accuracy')
    plt.title('accuracy曲线')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + name + '/' + name + '_Accuracy.png')
    plt.close()
# 不同数据集分析-组合数据（选取最优模型-对比不同特征提取的结果）
top_3 = '第三步'
os.makedirs('result/'+top_3, exist_ok=True)
data = pd.read_csv('./data/组合数据/1+2+3.csv',header=None)
# x,y
x = data[data.columns[1:]].values
y = data[data.columns[0]].values
# 降维分析
from sklearn.decomposition import PCA
PCA = PCA(n_components=0.98)
PCA_x = PCA.fit_transform(x)
print('降维的维度为：',PCA.n_components_)
import joblib
joblib.dump(PCA, os.getcwd() + '/result/' + top_3 + '/PCA.joblib')
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(PCA_x, y, test_size=0.2, random_state=123)
# 输入格式
X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print('X_train.shape, y_train.shape, X_test.shape, y_test.shape')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# 标签编码化
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
# 训练模型
model = Deep_learning_model('CNN_LSTM')
# 模型训练
history = model.fit(X_train, Y_train, epochs=50, batch_size=8,validation_data=(X_test, Y_test), verbose=2)
result = model_evaluation(y_test, np.argmax(model.predict(X_test),axis=1), model.predict(X_test), 'Val_CNN_LSTM')
result.to_csv(os.getcwd() + '/result/' + top_3 + '/CNN_LSTM_Val_result.csv')
model.save(os.getcwd() + '/result/' + top_3 + '/CNN_LSTM_Model.h5')
loss(history,top_3)
acc(history, top_3)
#########################################################
# 加载模型
max_model = load_model(os.getcwd() + '/result/' + top_3 + '/CNN_LSTM_Model.h5')
# 加载测试集
test = pd.read_csv('./data/测试数据/test1+2+3.csv',header=None)
test_x, test_y = test[test.columns[1:]].values, test[test.columns[0]].values
# 降维，加载PCA模型
loaded_pca = joblib.load(os.getcwd() + '/result/' + top_3 + '/PCA.joblib')
test_X  = loaded_pca.transform(test_x)
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('test_X.shape, test_y.shape')
print(test_X.shape, test_y.shape)
# 测试集评估
test_result = model_evaluation(test_y, np.argmax(max_model.predict(test_X),axis=1), max_model.predict(test_X), 'Test_CNN_LSTM')
test_result.to_csv(os.getcwd() + '/result/' + top_3 + '/CNN_LSTM_Test_result.csv')
ROC(['Training','Valing','Testing'], [model.predict(X_train),model.predict(X_test),max_model.predict(test_X)],
    [y_train,y_test,test_y],['navy','crimson','gold'], top_3)
PRC(['Training','Valing','Testing'], [model.predict(X_train),model.predict(X_test),max_model.predict(test_X)],
    [y_train,y_test,test_y],['navy','crimson','gold'], top_3)
# 验证集混淆矩阵
from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(y_test, model.predict(X_test).argmax(axis=1))
plt.savefig(os.getcwd() + '/result/' + top_3 + '/' + top_3 + '_验证集_混淆矩阵.png')
plt.close()
# 测试集混淆矩阵
plot_confusion_matrix(test_y, max_model.predict(test_X).argmax(axis=1))
plt.savefig(os.getcwd() + '/result/' + top_3 + '/' + top_3 + '_测试集_混淆矩阵.png')
plt.close()

###################################多模型对比
# 添加已经做好的模型CNN_LSTM
names = ['CNN_LSTM']
models = [max_model]
colors = ['darkorange','green','navy','crimson','orange','gold','mediumpurple']#,'steelblue'
# all_results = result.copy()
all_results = pd.read_csv(os.getcwd() + '/result/' + top_3 + '/CNN_LSTM_Val_result.csv',index_col=0)
# 训练其他模型
for i_model in ['CNN', 'LSTM', 'RNN', 'RF', 'SVM','KNN']:
    model = Deep_learning_model(i_model)
    # 模型训练
    if i_model == 'RF' or i_model == 'SVM' or i_model == 'KNN':
        model.fit(x_train, y_train)
        i_result = model_evaluation(y_test, model.predict(x_test), model.predict_proba(x_test), i_model)
    else:
        i_history = model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_data=(X_test, Y_test), verbose=2)
        i_result = model_evaluation(y_test, np.argmax(model.predict(X_test), axis=1), model.predict(X_test),i_model)  # i_model
    all_results = pd.concat([all_results, i_result])
    names.append(i_model)
    models.append(model)
all_results.to_csv(os.getcwd() + '/result/' + top_3 + '/模型对比_result.csv',index=False)
# 模型对比 AUC曲线、PRC曲线
# 绘制AUC\PRC
ROC_AUC(names,models,colors,top_3)
PRC_curve(names, models, colors,top_3)