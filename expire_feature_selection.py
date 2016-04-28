# -*- ecoding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame,Series  
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

def std_analysis(data, cell_number):
    '''
    从数据中提取出标准差特征
    '''
    dim_t = data.shape[0]
    data_c = data[:,1:];    # 只含数据的部分, 去掉第一列的时间
    std_t = np.zeros( (dim_t,4) )  # 第一列是时间，其他3列是对应的标准差

    for i in np.arange(dim_t):
        std_t[i,0] = data[i,0]
        for j in np.arange(1,4):  # 对3个不同的物理量计算标准差
            std_t[i,j] = np.std( data_c[i, (j-1) * cell_number : j * cell_number] )

    std_t_D = np.diff(std_t,1, axis=0);

    return  std_t, std_t_D

def get_training_data(feature_lin_var, feature_lin_lambda, data_exp, sample_num=8):
    '''
    构造进行线性回归的X和Y 
    '''
    sample_num = 8
    x_input = np.zeros( (sample_num, 6) )  
    y_input = np.zeros( (sample_num,) )

    for i in np.arange(sample_num):
        if i == 0 :
            tmp_index = 0
        elif i == sample_num-1 :
            tmp_index = i-2
        else:
            tmp_index = i-1

        x_input[i, :3] = feature_lin_var[tmp_index,:]
        x_input[i, 3:] = feature_lin_lambda[tmp_index,:]
        y_input[i] = data_exp[i,1]

    return (x_input, y_input)


# 从mat文件中提取需要的变量
data_pca_mat = sio.loadmat('data/pca_data.mat')
data_exp_mat = sio.loadmat('data/data_exp.mat')

data_all_cell_number = data_pca_mat['data_all_cell_number']
data_all_cell = data_pca_mat['data_all_cell']
data_index = data_pca_mat['data_index'].copy()
data_index[:,0] =  data_index[:,0]-1  # matlab 和 python 的下标起点不同
score = data_pca_mat['score'] 
data_exp = data_exp_mat['data_exp']

# linear regression

# 从不同的数据中提取出标准差特征
num_simulations = data_all_cell_number.shape[0]
feature_lin_var = np.zeros( (num_simulations, 3))
for i  in np.arange(num_simulations):
    data = data_all_cell[0,i]
    cell_number = data_all_cell_number[i,0]
    (std_t, std_t_D) = std_analysis( data , cell_number)
    for j in np.arange(3):
        feature_lin_var[i,j] = 1 / np.mean( std_t_D[:, j+1] /  std_t_D[:, 0]  )

# 由第1主分量特征构造特征
feature_lin_lambda = np.zeros( (num_simulations, 3))

for i  in np.arange(num_simulations):
    for j in np.arange(3):
        tmp_score = score[data_index[i,0] : data_index[i,1], 0, j]  # 第j个物理量的第1个主特征
        tmp_score_D = np.diff(tmp_score, 1, axis=0)
        tmp_time_D = np.diff(data_all_cell[0,i][:,0], 1, axis = 0)   # 数据的时间点并不是均匀的
        feature_lin_lambda[i,j] = 1 / np.mean( np.diff(tmp_score, 1, axis = 0) / tmp_time_D)


(x_input, y_input) = get_training_data(feature_lin_lambda=feature_lin_lambda, feature_lin_var=feature_lin_var, data_exp=data_exp)

# 对属性进行归一化
x_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))  

###  y 需不需要进行归一化？没有归一化的理由，但影响结果！！！
# y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))    
# x_input_minmax = x_scaler.fit_transform(x_input)
# y_input_minmax = y_scaler.fit_transform(y_input.reshape(-1,1))
# y_input_minmax = y_input_minmax.reshape((len(y_input_minmax)))
# 通过交叉验证来选择C
best_cv_score = -1e+30;
for log2c in np.arange(-10,30,1):
    clf = LinearSVR(C=2**log2c, epsilon=0.0001)
    clf.fit(x_input_minmax, y_input)
    cv_score = cross_val_score(cv=sample_num, estimator=clf, X=x_input_minmax, y=y_input, scoring= 'mean_squared_error').mean() # 留1
    print(cv_score)
    if cv_score > best_cv_score:
        best_cv_score = cv_score
        bestc = 2**log2c


# 利用所选的参数进行预测
clf = LinearSVR(C=bestc, epsilon=0.0001)
clf.fit(x_input_minmax, y_input)
y_pred = clf.predict(x_input_minmax)
# y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1))

view_point = 5;
plt.plot(x_input[:,view_point], y_input, 'bo-', x_input[:,view_point], y_pred, 'rs-')
plt.grid(True)
plt.legend(['y', 'y_pred'])
plt.show()









