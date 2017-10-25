import numpy as np
import h5py
from sklearn import svm
import math

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['x'][:]
        Y = hf['y'][:]
    return X, Y

x,y = load_h5py('Data/data_3.h5')

print y


max=-1
maxindex=-1
res=np.zeros(x.shape[0],np.amin(y)-np.amax(y)+1)
for i in range(np.amin(y),np.amax(y)+1):

    print i
    y_train=np.zeros(y.shape)
    for j in range(0,y.shape[0]):
        if(y[j]==i):
            y_train[j]=1
        else:
            y_train[j]=0
#    print y_train
    model=svm.SVC(kernel='linear')
    model.fit(x,y_train)
    m=model.coef_
    res.append(np.dot(x,m.T))

print res[0]

for i in range(0,x.shape[0]):
