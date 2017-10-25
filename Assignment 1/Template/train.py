import os
import os.path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y



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

        acr = float((acr_predict * 100) / len(traindata_valid_x))
        s_acr = s_acr + acr;
    s_acr = s_acr / k_f
    return s_acr


X,Y = load_h5py(args.train_data)

train_len = len(X)
train_len = int(train_len*0.8)
x_train=X[:train_len,:]
y_train=Y[:train_len,:]

testdata_x = X[train_len:, :]
testdata_y = Y[train_len:, :]



convertedYtrain=[]
row, col=y_train.shape
#print row, col
for i in range(0, row):
    for iter in range(0, col):
        if y_train[i][iter]==1:
            convertedYtrain.append(iter)
            break;

convertedYtrain=np.array(convertedYtrain)

convertedYtest=[]
row, col=testdata_y.shape
for i in range(0, row):
    for iter in range(0, col):
        if testdata_y[i][iter]==1:
            convertedYtest.append(iter)
            break;

convertedYtest=np.array(convertedYtest)

k_f=10

# Preprocess data and split it

# Train the models

if args.model_name == 'GaussianNB':
        model = GaussianNB()
        result = k_c_validate(k_f=k_f, training_model=model, traindata_x=x_train, traindata_y=convertedYtrain)
        model.fit(x_train, convertedYtrain)
        predicted_data_y=model.predict(testdata_x)
        crctpred = 0.000

        for k in range(0, len(convertedYtest)):
            if (np.array_equal(predicted_data_y[k], convertedYtest[k])):
                crctpred = crctpred + 1

        acr = float((crctpred * 100) / len(convertedYtest))
        print "Accuracy="+str(acr)
        weightfile = args.weights_path + "GaussianNB.sav"
        joblib.dump(model, weightfile)

        pass
elif args.model_name == 'LogisticRegression':
    params=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_val=-1
    iter=0
    best_index=-1
    result = [0 for i in range(0, len(params))]
    for i in params:
        model = LogisticRegression(C=i)
        result[iter] = k_c_validate(k_f=k_f, training_model=model, traindata_x=x_train, traindata_y=convertedYtrain)
        if (result[iter] > best_val):
            best_index = iter
            best_val = result[iter]
        iter = iter + 1

    best_model = LogisticRegression(C=params[best_index])

    best_model.fit(x_train, convertedYtrain)

    predicted_data_y = model.predict(testdata_x)

    crctpred = 0.000

    for k in range(0, len(convertedYtest)):
        if (np.array_equal(predicted_data_y[k], convertedYtest[k])):
            crctpred = crctpred + 1

    acr = float((crctpred * 100) / len(convertedYtest))
    print "Accuracy="+str(acr)
    weightfile = args.weights_path + "LogisticReg.sav"
    joblib.dump(best_model, weightfile)

    plt.plot(params, result)
    plt.savefig(args.plots_save_dir + "LogisticRegB.png")




elif args.model_name == 'DecisionTreeClassifier':
    params = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    best_val=-1
    iter=0
    best_index=-1
    result = [0 for i in range(0, len(params))]
    for i in params:
        model = DecisionTreeClassifier(max_depth=i,criterion="entropy")
        result[iter] = k_c_validate(k_f=k_f, training_model=model, traindata_x=x_train, traindata_y=convertedYtrain)
        if (result[iter] > best_val):
            best_index = iter
            best_val = result[iter]
        iter = iter + 1

    best_model= DecisionTreeClassifier(max_depth=params[best_index], criterion="entropy")
    best_model.fit(x_train, convertedYtrain)
    predicted_data_y = best_model.predict(testdata_x)

    crctpred = 0.000

    for k in range(0, len(convertedYtest)):
        if (np.array_equal(predicted_data_y[k], convertedYtest[k])):
            crctpred = crctpred + 1

    acr = float((crctpred * 100) / len(convertedYtest))
    print "Accuracy="+str(acr)
    weightfile = args.weights_path+"Decision_Tree.sav"
    joblib.dump(best_model,weightfile)

    plt.plot(params, result)
    plt.savefig(args.plots_save_dir+"Decision_treeB.png")





else:
	raise Exception("Invald Model name")

