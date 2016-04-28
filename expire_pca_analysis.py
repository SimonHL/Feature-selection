# -*- ecoding: utf-8 -*- 

# '''
# 根据分布来进行分析。 原Matlab版本直接使用的是空间上的分布， 本版本使用物理量取值的分布
# 原始数据格式： 周期数1  单元1的数据（数据1 数据2 数据3）
#                周期数1  单元2的数据（数据1 数据2 数据3）
#                ...
#                周期数1  单元N的数据（数据1 数据2 数据3）
#                周期数2  单元1的数据（数据1 数据2 数据3）
#                周期数2  单元2的数据（数据1 数据2 数据3）
#                ...
# 目标数据格式： 共有6次仿真结果，因周期数不同，存放到List中   data_simulations
#                每个仿真结果包括4块数据： 周期数  特征1数据（从单元1到单元N）   特征2数据   特征3数据
# '''
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

def get_data_from_simulation(filename, cell_number) :
    '''
    从一次仿真数据中整理出便于处理的格式
    Outputs:
        t_span: (time_number,)
        new_data: (time_number, cell_number, 3)  3对应3个不同的物理量                  
    '''
    tmp_data = np.load(filename+'.npy')
    t_span = np.unique(tmp_data[:,0])
    new_data = np.zeros((len(t_span), cell_number, 3))
    for i in np.arange(3):
        for t in np.arange(len(t_span)):
            new_data[t,:,i] = tmp_data[t * cell_number: (t+1)*cell_number, i+2]
    return  (t_span, new_data)

def get_hist(data, hist_bins, range_bins):
    '''
    根据仿真结果得到随时间变化的概率分布
    Inputs:
        data:  (t_number, cell_number, dim_pysical) 仿真数据块 
        range_bins: (dim_pysical, 2)  3个不同物理量对应的最小值和最大值
    Outputs:
        data_hist (t_number, hist_bins, dim_pysical) 将cell_number的数据转化为hist
    '''
    t_number = data.shape[0]
    data_hist = np.zeros((t_number, hist_bins, 3))
    for i in range(3):
        for t in np.arange(t_number):
            hist_tmp,_ = np.histogram(data[t,:,i], 
                                      bins=hist_bins, 
                                      range=range_bins[i])
            hist_tmp = hist_tmp / hist_tmp.sum()
            data_hist[t,:, i] = hist_tmp

    return data_hist

def get_exp_data(data_list, cell_number=27000):
    '''
    从仿真数据中找出对应寿命点的数据
    因为采样间隔的精度问题，0045的两次实验对应的数据， 从更精细化的数据文件中取得
    Inputs:
        data_list: 仿真数据
                   仿真次数 len(data_list)
                   数据格式  (times, cell_number, dim_pysical), 注意，每次仿真的times不同
    Returns: 
        data_exp_time: (exp_num,)
        data_exp_data: (exp_num, cell_number, dim_pysical)

    '''
    # 暂时不用的数据
    # data_exp_stress_list = [  45,   45,   60,  80,  90, 100, 130, 130] # 实验所用的应力
    # data_exp_true_list   = [9904, 5457, 1494, 370, 207, 354,  94,  80] # 实验所得的周期数，找对应数据时有一点误差

    filenames_exp = ['data/data_for_zmh0045-exp-9904',  'data/data_for_zmh0045-exp-5457']

    # 不同实验的对应数据来源不同
    data_exp_fromdata_list =[get_data_from_simulation(filenames_exp[0], cell_number), 
                         get_data_from_simulation(filenames_exp[1], cell_number),
                        data_list[1],
                        data_list[2],
                        data_list[3],
                        data_list[4],
                        data_list[5],
                        data_list[5] ]

    # 数据在数据源中的位置
    data_exp_fromdata_index_list = [70, 36, 168, 42, 24, 40, 11, 10]
    data_exp_fromdata_index_list = [i-1 for i in data_exp_fromdata_index_list] # 下标从0开始

    experiment_num = len(data_exp_fromdata_list)

    data_exp_time = np.zeros((experiment_num,))
    data_exp_data = np.zeros((experiment_num, cell_number, 3))
    for i in range(experiment_num):
        data_ = data_exp_fromdata_list[i]
        index_ = data_exp_fromdata_index_list[i]
        data_exp_time[i] = data_[0][index_]
        data_exp_data[i,:,:] = data_[1][index_,:,:]

    return data_exp_time, data_exp_data

def prepare_data_for_pca(data_list, data_exp_data, use_hist=True):
    '''
    准备PCA分析所要用到的数据
    Inputs:
        data_list ： 仿真数据 
        data_exp_data： 实验数据对应的数据
    Returns： 
        pca_data_all ： 全部数据
        pca_data_exp ： 实验点对应的数据
        pca_data_list : 不同的仿真过程
    '''

    data_all = np.concatenate([data_list[i][1] for i in range(len(data_list))], axis=0)
    data_list_pure = [data_list[i][1]  for i in range(len(data_list)) ] # 只使用实验数据

    if use_hist:
        # 3个物理量的最小值和最大值
        data_all_minmax_list = [(data_all[:,:,i].min(), data_all[:,:,i].max()) for i in range(3)]
        pca_data_list = [get_hist(_tmp,hist_bins, range_bins=data_all_minmax_list) for _tmp in data_list_pure]
        pca_data_all = np.concatenate(pca_data_list, axis=0)
        pca_data_exp = get_hist(data_exp_data, hist_bins, range_bins=data_all_minmax_list)  
    else:
        pca_data_all = data_all
        pca_data_exp = data_exp_data
        pca_data_list = data_list_pure

    return pca_data_all, pca_data_exp, pca_data_list

def pca_analysis(pca_data_all, pca_data_exp, pca_data_list, dim_choose=0):
    '''
    PCA分析 
    Inputs:
         pca_data_all: 用于训练
         pca_data_exp： 用于生成实验数据点
         pca_data_list 用于生成仿真过程数据
         dim_choose： 选择进行分析的物理量
    Outputs:
    '''
    pca = PCA()
    pca_data_all_transformed = pca.fit_transform(pca_data_all[:,:,dim_choose]) # 训练

    pca_result_list = []
    for i in range(len(pca_data_list)):
        pca_data = pca.transform(pca_data_list[i][:,:,dim_choose])
        pca_result_list.append(pca_data)

    pca_result_exp = pca.transform(pca_data_exp[:,:,dim_choose])

    return pca_result_list, pca_result_exp


# 主程序 Begin
cell_number = 27000  # 单元数固定为27000
hist_bins = 100  # 进行密度估计时的样条个数
filenames = [
    'data/data_for_zmh_0045', 
    'data/data_for_zmh_006', 
    'data/data_for_zmh_008',
    'data/data_for_zmh_009',
    'data/data_for_zmh_010',
    'data/data_for_zmh_013'
]

# 仿真数据
data_list = [get_data_from_simulation(f, cell_number) for f in filenames]

# 寿命对应的数据
data_exp_time, data_exp_data = get_exp_data(data_list, cell_number)

#  直方图，或直接使用原始数据
pca_data_all, pca_data_exp, pca_data_list = prepare_data_for_pca(data_list, data_exp_data, use_hist=True)

# PCA 分析
pca_result_list, pca_result_exp = pca_analysis(pca_data_all, pca_data_exp, pca_data_list, dim_choose=0)

#############  绘图部分   ####################
view_point = 0    # 需要观察的主分量
    
# 实测数据的位置
pos = [(data_exp_time[i],  pca_result_exp[i,view_point]) for i in range(len(data_exp_time))]
pos = np.asarray(pos)

# 绘图
for i in range(len(pca_result_list)):
    t = data_list[i][0]  # 仿真的时间
    y = pca_result_list[i][:,view_point]
    plt.plot(t, y)

plt.plot(pos[:,0], pos[:,1], 'ro')
plt.grid(True)
plt.show()







