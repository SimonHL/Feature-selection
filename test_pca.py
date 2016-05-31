# -*- ecoding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# t = np.arange(0,100,0.5)
# k = np.arange(1,5) + 
# t.resize(1,len(t))
# k.resize(len(k),1)

# x = np.dot(k,t)
a = 10  # 用来定义长轴的长度

x = np.random.uniform(low=-1,high=1, size=(1000,2))
x[:,0] = a * x[:,0]

x=x[x[:,0]**2 / a**2 + x[:,1]**2 < 1]  # 选择一个椭圆部分

plt.plot(x[:,0], x[:,1], 'ro')

x_center = np.asarray([2.0,1.0]).reshape((1,2))
x = x + x_center
sqrt_3 = np.sqrt(3)
V = np.asarray([ [sqrt_3 /2, 1/2],[-1/2, sqrt_3/2]])

y = np.dot(x,V)
plt.plot(y[:,0], y[:,1], 'bo')

pca = PCA()

new_y = pca.fit_transform(y)

plt.plot(new_y[:,0], new_y[:,1], 'yo')
print(pca.explained_variance_)

# plt.ylim((-a,a))
plt.grid(True)
plt.legend(['a','b', 'c'])
plt.show()





