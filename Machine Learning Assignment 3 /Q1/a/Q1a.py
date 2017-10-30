

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import math
from sklearn.utils import shuffle


def k_c_validate(k_f, training_model, traindata_x, traindata_y):
        s_acr = 0.000000
        model = training_model
        maxacr=-1
        for i in range(k_f):
            print i
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

            training_model.fit(traindata_training_x, traindata_training_y,8)

            traindata_valid_x = np.array(traindata_valid_x)
            traindata_valid_y = np.array(traindata_valid_y)

            acr_predict = 0

            # for k in range(0, len(train_data_val_x)):
            #   original=train_data_val_y


            acr = (nn.score(traindata_valid_x,traindata_valid_y) * 100)
            print acr
            if(acr>maxacr):
                model=training_model
            s_acr = s_acr + acr;
        joblib.dump(model,'Model1/logistic_0_9.sav')
        s_acr = s_acr / k_f
        print s_acr
        return  s_acr


def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return x * (1.0 - x)



class NN:

    def __init__(self,_layer,learning_rate):
        self.layer = list(_layer)
        self.learningrate = learning_rate
        self.weight=[]
        self.basis=[]


    def fit(self,x,y,it):
        xshape = x.shape
        yshape=y.shape
        print x.shape
        activation=[]
        self.weight=[]
        self.basis=[]


        '''first layer weight and basis'''
        self.weight.append(np.random.uniform(low=-1.0,high=1.0,size=(xshape[1],self.layer[0])))
        #print self.weight[-1].shape
        self.basis.append(np.random.uniform(low=-1.0,high=1.0, size=(1,self.layer[0])))

        '''second layer onwards weight & basis initialisation'''
        for i in range(1,len(self.layer)):
            xtemp = np.random.uniform(low=-1.0, high=1.0, size=(self.layer[i-1], self.layer[i]))
            self.basis.append(np.random.uniform(low=-1.0,high=1.0, size=(1,self.layer[i])))
            self.weight.append(xtemp)
         #   print self.weight[-1].shape

        '''output layer weight and basis initialisation'''
        #last weight 50x1
        self.weight.append(np.random.uniform(low=-1.0, high=1.0, size=(self.layer[-1],1)))
        self.basis.append(np.random.uniform(low=-1.0, high=1.0, size=(1,1)))
        #print self.weight[-1].shape

        '''weight matrix list order 784x100, 100x50, 50x1'''


        for m in xrange(0,it):
            for r in range(0,x.shape[0]):
                x_input = x[r]/float(255.0)
                activation=[]
                # print x[i].shape
                activation.append(x_input)
                '''calculate activation values for neurons and append to activation list'''
                for l in range(len(self.layer)+1):
                    # print activation[-1].shape
                    activation.append(sigmoid(np.dot(activation[-1],self.weight[l])+self.basis[l]))
                    # print self.weight[l].shape
                val = activation[-1]
                # print val.shape
                error_o = (val-y[r])*dsigmoid(val)

                errors=[]
                errors.append(error_o)


                # print len(activation)
                for i in range(len(activation)-2,0,-1):
                     # print i
                     errors.append(errors[-1].dot(self.weight[i].T)*dsigmoid(activation[i]))


                for i in range(0,len(self.weight)):
                    # print len(self.weight)
                    self.weight[i]=self.weight[i]-self.learningrate*np.dot(np.atleast_2d(activation[i]).T,errors[-(i+1)])
                    '''basis updation'''
                    self.basis[i]=self.basis[i]-np.sum(errors[-(i+1)])*self.learningrate
            # print 'Error at iteration'+str(m)+ ':'+str(error_o)+'\n'




    def predict (self,xtest):
        pred=[]
        for i in range(xtest.shape[0]):
            activation=[]
            activation.append(xtest[i]/255.0)
            for l in range(len(self.layer) + 1):
                activation.append(sigmoid(np.dot(activation[-1], self.weight[l]) + self.basis[l]))

            val=activation[-1]
            # print activation[-1].shape
            if val>=0.5:
                pred.append(0)
            else:
                pred.append(1)

        # print self.weight
        return pred

    def score(self,xtest,ytest):
        pred = []
        for i in range(xtest.shape[0]):
            activation = []
            activation.append(xtest[i]/255.0)
            for l in range(len(self.layer) + 1):
                activation.append(sigmoid(np.dot(activation[-1], self.weight[l]) + self.basis[l]))

            val = activation[-1]

            if val >= 0.5:
                pred.append(1)
            else:
                pred.append(0)
        sum=0
        for i in range(0,ytest.shape[0]):
            if(ytest[i]==pred[i]):
                sum=sum+1

        return sum/float(ytest.shape[0])



def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y

X,Y = load_h5py('/home/jarvis/Desktop/Machine Learning Assignment 3 (2)/Dataset/dataset_partA.h5')
x_f=[]
for i in range(0,X.shape[0]):
    x_f.append(X[i].flatten())

x_f=np.array(x_f)


for i in range(0,Y.shape[0]):
    if(Y[i]==7):
        Y[i]=0
    else:
        Y[i]=1


'''

nn=NN((100,50),0.9)
print k_c_validate(k_f=5,training_model=nn,traindata_y=Y,traindata_x=x_f)

# learning_rate=[0.1,0.3,0.5,0.7,0.9]
# epoch=[2,4,6,8,10]
# acr=[]
# for i in alpha:
#
#     nn=NN((100,50),i)
#
#     acr.append(k_c_validate(k_f=4,training_model=nn,traindata_x=x_f,traindata_y=Y))
# print  acr
# acr_mat = np.array(acr)
# epoch_mat = np.array(alpha)
#
# plt.plot(epoch_mat,acr_mat)
# plt.ylabel('Accuracy')
# plt.xlabel('Alpha')
# plt.title('Sigmoid NN')
# plt.show()
#
# print alpha
# print acr
'''

nn=joblib.load('logistic_0_9.sav')
print nn.score(x_f,Y)







'''References
1.Machine Learning an Algorithmic Perspective-Marsland
2.Andrew Ng's Notes
Discussed Concepts with Vaibhav Varshney
'''
