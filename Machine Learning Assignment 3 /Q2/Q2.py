import numpy as np
import h5py
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import idx2numpy
from sklearn.utils import shuffle
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

        training_model.fit(traindata_training_x, traindata_training_y)

        traindata_valid_x = np.array(traindata_valid_x)
        traindata_valid_y = np.array(traindata_valid_y)

        acr_predict = 0

        # for k in range(0, len(train_data_val_x)):
        #   original=train_data_val_y
        predicted = training_model.predict(traindata_valid_x)

        for k in range(0, len(traindata_valid_y)):
            if (np.array_equal(predicted[k], traindata_valid_y[k])):
                acr_predict = acr_predict + 1

        acr = (acr_predict * 100) / float(len(traindata_valid_x))
        s_acr = s_acr + acr;
    s_acr = s_acr / k_f
    return s_acr



def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y
#
X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
Y =idx2numpy.convert_from_file('train-labels.idx1-ubyte')

X_t = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
Y_t =idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# print Y.shape
'''
X, Y = load_h5py('dataset_partA.h5')
x_f = []
for i in range(0, X.shape[0]):
    x_f.append(X[i].flatten())

x_f = np.array(x_f)
# for i in range(0, Y.shape[0]):
#     if (Y[i] == 7):
#         Y[i] = 0
#     else:
#         Y[i] = 1

# x_f,Y=shuffle(x_f,Y,random_state=0)
# nn=MLPClassifier(hidden_layer_sizes=(100,50),verbose=True,activation='logistic',alpha=0.1,early_stopping=True)
# nn.fit(x_f[:10000],Y[:10000])
# print nn.score(x_f[10000:],Y[10000:])
'''
x_t=[]

for i in range(0,X_t.shape[0]):
    x_t.append(X_t[i].flatten())

x_t=np.array(x_t)


x_f=[]

for i in range(0,X.shape[0]):
    x_f.append(X[i].flatten())

x_f=np.array(x_f)
print x_f.shape
#
#
# #
# # #
# nn = MLPClassifier(hidden_layer_sizes=(100,50), activation="logistic", verbose=True, early_stopping=True,alpha=0.15)
# #
# # nn.fit(x_f,y=Y)
# validation_score = nn.validation_scores_
# print validation_score

nn= joblib.load('Model2b/logistic0_15.sav')
print nn.score(x_t,Y_t)
# joblib.dump(nn,'Model2a/logistic0_1.sav')