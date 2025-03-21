import os
import warnings
import pandas as pd
import numpy as np
from keras.layers import *
from keras.models import *
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from evaluation import model_evaluation
warnings.filterwarnings("ignore")
# 设置中文支持并增大文字大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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
    return model
# loss损失值
def loss(history,name):
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train-loss')
    plt.plot(history.history['val_loss'], label='test-loss')
    plt.legend()
    plt.title('loss曲线')
    plt.savefig(os.getcwd() + '/result/' + top_1 + '/' + name + '_loss.png')
    plt.close()
# acc准确率
def acc(history,name):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train-accuracy')
    plt.plot(history.history['val_accuracy'], label='test-accuracy')
    plt.title('accuracy曲线')
    plt.legend()
    plt.savefig(os.getcwd() + '/result/' + top_1 + '/' + name + '_Accuracy.png')
    plt.close()
# 不同数据集分析-单独数据（选取优化模型）
top_1 = '第一步'
os.makedirs('result/'+top_1, exist_ok=True)
all_results = pd.DataFrame()
# all_results = all_results.loc[~all_results.index.isin(['2.csv'])]
for data_name in os.listdir('./data/单独数据/'):#
    data = pd.read_csv('./data/单独数据/'+data_name,header=None)
    # x,y
    x = data[data.columns[1:]].values
    y = data[data.columns[0]].values
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
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
    loss(history, data_name)
    acc(history, data_name)
    # 评估模型
    result = model_evaluation(y_test, np.argmax(model.predict(X_test),axis=1), model.predict(X_test), 'CNN_LSTM')
    result.index = [data_name]
    all_results = pd.concat([all_results,result])
    print(all_results)
# ll = pd.DataFrame()
# for i in os.listdir('./data/单独数据/'):
#     ls = all_results.loc[all_results.index.isin([i])]
#     ll = pd.concat([ll,ls])
# all_results = ll
all_results.to_csv(os.getcwd() + '/result/' + top_1 + '/result.csv')



