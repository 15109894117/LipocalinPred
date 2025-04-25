import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.utils.np_utils import to_categorical
from evaluation import model_evaluation,ROC,PRC
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
# 设置中文支持并增大文字大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 构建模型
def Deep_learning_model(optimizer_name):
    model = Sequential()
    model.add(
        Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])
    return model
# 不同数据集分析-组合数据（选取最优模型-对比不同特征提取的结果）
top_4 = '第四步'
os.makedirs('result/'+top_4, exist_ok=True)
data = pd.read_csv('./data/组合数据/1+2+3.csv',header=None)
# x,y
x = data[data.columns[1:]].values
y = data[data.columns[0]].values
# 降维分析
from sklearn.decomposition import PCA
PCA = PCA(n_components=0.98)
PCA_x = PCA.fit_transform(x)
import joblib
joblib.dump(PCA, os.getcwd() + '/result/' + top_4 + '/PCA.joblib')
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
list_proba = []
list_loss = []
list_acc = []
all_results = pd.DataFrame()
optimizer_name = ['adamax', 'nadam', 'adam','rmsprop','experimentaladagrad']#
for solver_name in optimizer_name:
    model = Deep_learning_model(solver_name)
    history = model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_data=(X_test, Y_test), verbose=2)
    result = model_evaluation(y_test, np.argmax(model.predict(X_test), axis=1), model.predict(X_test), solver_name)
    list_loss.append(history.history['val_loss'])
    list_acc.append(history.history['val_accuracy'])
    list_proba.append(model.predict(X_test))
    all_results = pd.concat([all_results, result])
print(all_results)
all_results.to_csv(os.getcwd() + '/result/' + top_4 + '/优化器对比_result.csv',index=False)
# 可视化LOSS\ACC对比
# loss
plt.figure()
for i_loss,i_name in zip(list_loss,optimizer_name):
    plt.plot(i_loss, label=i_name+' loss')
plt.legend()
plt.title('All_optimizer_loss')
plt.savefig(os.getcwd() + '/result/' + top_4 + '/优化器对比_loss.png')
plt.close()
# acc
plt.figure()
for i_acc,i_name in zip(list_acc,optimizer_name):
    plt.plot(i_acc, label=i_name+' accuracy')
plt.legend()
plt.title('All_optimizer_accuracy')
plt.savefig(os.getcwd() + '/result/' + top_4 + '/优化器对比_accuracy.png')
plt.close()
# 不同优化器对比的ROC曲线
ROC(optimizer_name,list_proba,y_test,['darkorange','green','crimson','mediumpurple','gold'],top_4)#,'navy'
# 不同优化器对比的PRC曲线
PRC(optimizer_name,list_proba,y_test,['darkorange','green','crimson','mediumpurple','gold'],top_4)#,'navy'