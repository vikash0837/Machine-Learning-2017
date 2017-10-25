import math
import  json
import sklearn
import matplotlib as plt
from pprint import pprint
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import svm
def load_file(filename):
    json_data = open(filename)
    return json.load(json_data)

data=load_file('Json/train.json')
from sklearn.feature_extraction import  DictVectorizer

print len(data)

# print count
Y=[]
X=[]

test=[]

for i in range(0,len(data)):
    Y.append(data[i]['Y'])
    X.append(data[i]['X'])
    test.append(X[i])
    test[i].append(Y[i])

print test[0]

test=np.unique(test)


X=[]
Y=[]

print test[3][-1:]

for i in range(0,test.shape[0]):
    X.append(test[i][:-1])
    Y.append(test[i][-1:])

for i in range(0,len(X)):
    X[i]=np.array_str(np.array(X[i]))[1:-1]

Y=np.array(Y).flatten()
print len(X)
print Y.shape
vect = CountVectorizer(analyzer='word',ngram_range=(1,2),token_pattern='\d+')
X_train = vect.fit_transform(X)
print len(vect.get_feature_names())
print X_train.shape

parameters={'kernel':('linear', 'rbf'), 'C':[1,5]}
# for i in np.arange(5,5.3,0.3):
#     clf = svm.LinearSVC(C=i)
#     clf.fit(X_train,Y)
#     print i
#     print clf.score(X=X_train,y=Y)

clf = svm.SVC(kernel='poly',degree=3)
clf.fit(X_train,Y)
print clf.score(X_train,Y)
#
joblib.dump(clf,'Models/linear_svmpoly_c3.sav')
# y_pred = logreg.predict(X_train)
#
# count=0
# for i in range(0,len(y_pred)):
#     if(Y[i]==y_pred[i]):
#         count =count+1
#
# print count
# print count/float(len(y_pred))
# #
test_file=load_file('Json/test.json')
X_test=[]
for i in range(0,len(test_file)):
     X_test.append(test_file[i]['X'])
for i in range(0,len(X_test)):
     X_test[i]=np.array_str(np.array(X_test[i][:]))[1:-1]
X_test=vect.transform(X_test)

pred_y=clf.predict(X_test)

f=open('predictionpoly_c_3.csv','w')
f.write('Id,Expected'+'\n')

for i in range(0,len(pred_y)):
    f.write(str(i+1)+','+str(pred_y[i])+'\n')

f.close()


