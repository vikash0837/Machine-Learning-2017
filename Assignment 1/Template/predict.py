import os
import os.path
import argparse
import h5py
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--weights_path", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--output_preds_file", type=str)

args = parser.parse_args()


# load the test data
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y


X, Y = load_h5py(args.test_data)
if args.model_name == 'GaussianNB':
    model = joblib.load(args.weights_path)
    Y_predicted = model.predict(X)
    f = open(args.output_preds_file, 'w')

    for i in range(len(Y_predicted)):
        f.write(str(Y_predicted[i]) + '\n')

elif args.model_name == 'LogisticRegression':
    model = joblib.load(args.weights_path)
    Y_predicted = model.predict(X)
    f = open(args.output_preds_file, 'w')

    for i in range(len(Y_predicted)):
        f.write(str(Y_predicted[i]) + '\n')

elif args.model_name == 'DecisionTreeClassifier':
    model = joblib.load(args.weights_path)
    Y_predicted = model.predict(X)
    f = open(args.output_preds_file, 'w')

    for i in range(len(Y_predicted)):
        f.write(str(Y_predicted[i]) + '\n')
else:
    raise Exception("Invald Model name")

