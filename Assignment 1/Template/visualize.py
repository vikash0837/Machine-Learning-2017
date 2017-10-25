import os
import os.path
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y


X, Y = load_h5py(args.data)

x= np.array(X)
y= np.array(Y)

colorset=['red','orange','pink','green','black','gray','blue','brown','yellow','cyan']

color=[]
for i in range(0,len(y)):
    j=0
    for j in range(0,y.shape[1]):
        if(y[i][j]==1):
            break

    color.append(colorset[j])

X_embedded = TSNE(n_components=2,verbose=1).fit_transform(x)
plt.scatter(x=X_embedded[:,0],y=X_embedded[:,1],c=color)
plt.savefig(args.plots_save_dir + "_visualise.png")







