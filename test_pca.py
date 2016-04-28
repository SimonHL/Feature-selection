# -*- ecoding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

t = np.arange(0,100,0.5)
k = np.arange(1,5)
t.resize(1,len(t))
k.resize(len(k),1)

x = np.dot(k,t)
pca = PCA()

new_x = pca.fit_transform(x)
pca.explained_variance_



