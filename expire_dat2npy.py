# -*- ecoding: utf-8 -*-
'''
将txt格式的数据转化为二进制数据
'''
import numpy as np
import matplotlib.pyplot as plt 

# 读取数据  

filenames = ['data/data_for_zmh_0045', 
             'data/data_for_zmh_006',
             'data/data_for_zmh_008',
             'data/data_for_zmh_009',
             'data/data_for_zmh_010',
             'data/data_for_zmh_013']

cell_number = 27000  # 单元数固定为27000

for f in filenames:
    np.save(f+'.npy', np.loadtxt(f + '.dat'))

filenames = ['data/data_for_zmh0045-exp-5457',
             'data/data_for_zmh0045-exp-9904']

for f in filenames:
    np.save(f+'.npy', np.loadtxt(f + '.dat'))