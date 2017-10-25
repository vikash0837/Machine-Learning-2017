# import numpy as np
import math
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
def rbf_dist(x,y):
    return np.sum((x-y)**2)

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['x'][:]
        Y = hf['y'][:]
    return X, Y


X, Y = load_h5py('Data/data_1.h5')

x = np.array(X)
y = np.array(Y)

