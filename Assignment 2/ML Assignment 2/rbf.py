# import numpy as np
import math
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm



def predict(model,x,y,xtest):

    n_class = len(np.unique(Y))
    dual_var_list = []
    support_vec_list = []
    intercept_list = []

    for i in range(0,n_class):
        y_load=[]

        for j in range(0,len(y)):
            if(y[j]==i):
                y_load.append(1)
            else:
                y_load.append(0)
        y_load=np.array(y_load).flatten()
        model.fit(x,y_load)

        dual_var_list.append(model.dual_coef_)
        support_vector=[x[k] for k in model.support_]
        support_vec_list.append(support_vector)
        intercept_list.append(model.intercept_)

    #print len(support_vec_list[0])

    pred=[]
    for i in range(n_class):
        #print i

        coef = dual_var_list[i]
        support_vec = support_vec_list[i]
        #print len(support_vec), len(coef[0])
        #print support_vec
        intercept = intercept_list[i]

        k = rbfkernel(xtest,np.array(support_vec))
        #print k.shape
        pred.append(np.dot(k,coef.T)+intercept)
        pred[i]=np.array(pred[i]).flatten()

    #print pred[0][0],pred[1][0]

    pred_final=[]
    for i in range(0,len(xtest)):
        max = pred[0][i]
        maxindex=0
        for j in range(1,n_class):
            if pred[j][i]>max :
                max=pred[j][i]
                maxindex=j
        pred_final.append(maxindex)
    pred_final=np.array(pred_final).flatten()
    return pred_final


def k_c_validate(k_f, training_model, traindata_x, traindata_y):
    s_acr = 0.000000

    for i in range(k_f):
        k_size = (len(traindata_x)) / k_f

        traindata_valid_x = []
        traindata_training_x = []

        traindata_valid_y = []
        traindata_training_y = []

        for j in range(0, len(traindata_x)):
            if (int(j / k_size) == i):
                traindata_valid_x.append(traindata_x[j])
                traindata_valid_y.append(traindata_y[j])
            else:
                traindata_training_x.append(traindata_x[j])
                traindata_training_y.append(traindata_y[j])

        traindata_training_x = np.array(traindata_training_x)
        traindata_training_y = np.array(traindata_training_y)

        traindata_valid_x = np.array(traindata_valid_x)
        traindata_valid_y = np.array(traindata_valid_y)

        training_model.fit(traindata_training_x, traindata_training_y)

        # for k in range(0, len(train_data_val_x)):
        #   original=train_data_val_y
        predicted = predict(model, traindata_training_x, traindata_training_y, traindata_training_x)

        acr_predict = 0

        for k in range(0, len(traindata_valid_y)):
            if (np.array_equal(predicted[k], traindata_valid_y[k])):
                acr_predict = acr_predict + 1

        acr = float((acr_predict * 100) / len(traindata_valid_x))
        s_acr = s_acr + acr;
    s_acr = s_acr / k_f
    return s_acr


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

def rbfkernel(X,Y):

    k=np.zeros(shape=(X.shape[0],Y.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(0,Y.shape[0]):
            k[i][j]=rbf_dist(X[i,:],Y[j,:])

    for i in range(0,X.shape[0]):
        for j in range(0,Y.shape[0]):
            k[i][j]=math.exp(-1*(0.9)*k[i][j])
    return k

model=svm.SVC(kernel=rbfkernel)

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = predict(model,x,y,np.c_[xx.ravel(), yy.ravel()])
print xx.shape
print Z.shape
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('Classification using Support Vector Machine with custom'
          ' kernel')
plt.axis('tight')
plt.show()


pred_final=predict(model,x,y,x)

pred_final=predict(model,x,y,x)
n_classes = len(np.unique(y))
matrix = np.zeros((n_classes, n_classes))
for i in range(0,len(x)):
    matrix[y[i],pred_final[i]]=matrix[y[i],pred_final[i]]+1
print matrix
plt.matshow(matrix)
plt.colorbar()
plt.show()





model1 = svm.SVC(kernel='rbf')
model1.fit(x,y)
pred_inblt = model1.predict(x)


count=0
for i in range(0,len(Y)):
    if(pred_final[i]==pred_inblt[i]):
        count=count+1
print count/float(len(Y))