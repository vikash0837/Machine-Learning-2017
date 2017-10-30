import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import math
from sklearn.utils import shuffle
import idx2numpy


def k_c_validate(k_f, training_model, traindata_x, traindata_y,epoch):
    s_acr = 0.000000
    max_scr=-1
    model=training_model
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

        training_model.fit(traindata_training_x, traindata_training_y,epoch)

        traindata_valid_x = np.array(traindata_valid_x)
        traindata_valid_y = np.array(traindata_valid_y)

        acr_predict = 0

        # for k in range(0, len(train_data_val_x)):
        #   original=train_data_val_y


        acr = (nn.score(traindata_valid_x,traindata_valid_y) * 100)
        print acr
        if(max_scr<acr):
            max_scr=acr
            model=training_model

        s_acr = s_acr + acr;
    joblib.dump(model,'Model1/relu_test.sav')
    s_acr = s_acr / k_f
    return s_acr


def reLu(x):
    x[x<0]=1
    return np.log(x)

def dreLu(x):
    for j in range(len(x)):
        for i in range(0,len(x[j])):
            if(x[0][i]>0):
                x[j][i]=1
            else:
                x[j][i]=0
    return x




def softmax(x):
    return np.exp(x) / float(np.sum(np.exp(x)))

def dsoftmax(x):
    return

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
        self.weight.append(np.random.uniform(low=-1.0, high=1.0, size=(self.layer[-1],2)))
        self.basis.append(np.random.uniform(low=-1.0, high=1.0, size=(1,2)))
        #print self.weight[-1].shape

        '''weight matrix list order 784x100, 100x50, 50x1'''


        for m in xrange(0,it):
            for r in range(0,x.shape[0]):
                x_input = x[r]/float(255.0)
                activation=[]
                # print x[i].shape
                activation.append(x_input)
                '''calculate activation values for neurons and append to activation list'''
                for l in range(len(self.layer)):
                    # print activation[-1].shape
                    activation.append(reLu(np.dot(activation[-1],self.weight[l])+self.basis[l]))
                    # print self.weight[l].shape
                activation.append(softmax(np.dot(activation[-1],self.weight[-1])+self.basis[-1]))
                val = activation[-1]
                # print val.shape
                error_o = (val-y[r])
                # print error_o
                errors=[]
                errors.append(error_o)


                # print len(activation)
                for i in range(len(activation)-2,0,-1):
                     # print i
                     errors.append(errors[-1].dot(self.weight[i].T)*dreLu(activation[i]))


                for i in range(0,len(self.weight)):
                    # print len(self.weight)
                    self.weight[i]=self.weight[i]-self.learningrate*np.dot(np.atleast_2d(activation[i]).T,errors[-(i+1)])
                    '''basis updation'''
                    self.basis[i]=self.basis[i]-np.sum(errors[-(i+1)])*self.learningrate
            print 'Error at iteration'+str(m)+ ':'+str((error_o)/2)+'\n'




    def predict (self,xtest):
        pred=[]
        for i in range(xtest.shape[0]):
            activation=[]
            activation.append(xtest[i]/float(255.0))
            for l in range(len(self.layer)):
                activation.append(reLu(np.dot(activation[-1], self.weight[l]) + self.basis[l]))

            activation.append(softmax(np.dot(activation[-1], self.weight[-1]) + self.basis[-1]))
            val = activation[-1]

            val=activation[-1]
            print val[0]
            # print activation[-1].shape
            temp = np.zeros(2)
            temp.reshape(1,2)
            temp[np.argmax(val[0])]=1
            pred.append(temp)
        # print self.weight
        return pred

    def score(self,xtest,ytest):
        pred = []
        sum = 0
        for i in range(xtest.shape[0]):
            activation = []
            activation.append(xtest[i]/255.0)
            for l in range(len(self.layer)):
                activation.append(reLu(np.dot(activation[-1], self.weight[l]) + self.basis[l]))
            activation.append(softmax(np.dot(activation[-1], self.weight[-1]) + self.basis[-1]))

            val = activation[-1]

            if(np.argmax(val[0])==np.argmax(ytest[i])):
                sum=sum+1


        return sum/float(ytest.shape[0])



def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y
# X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
# Y =idx2numpy.convert_from_file('train-labels.idx1-ubyte')
#
# X_t = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
# Y_t =idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
#
#
# x_t=[np.array(i.flatten()) for i in X_t]
# x_f=[np.array(i.flatten()) for i in X]
#
# x_t=np.array(x_t)
# x_f=np.array(x_f)
#
# x_t=[]
# for i in range(0,X.shape[0]):
#     x_f.append(X_t[i].flatten())


# y_t=[]
#
# for i in range(0,Y_t.shape[0]):
#     temp = np.zeros(10)
#     temp[Y[i]]=1
#     y_t.append(temp)
#
# for i in range(0,Y.shape[0]):
#     if(Y[i]==7):
#         Y[i]=0
#     else:
#         Y[i]=1
#
#
# y_f=[]
#
# for i in range(0,Y.shape[0]):
#     temp = np.zeros(10)
#     temp[Y[i]]=1
#     y_f.append(temp)
# y_f=np.array(y_f)
#
# y_t=[]
#
# for i in range(0,Y_t.shape[0]):
#     temp = np.zeros(10)
#     temp[Y_t[i]]=1
#     y_t.append(temp)
# y_t=np.array(y_t)


X,Y = load_h5py('dataset_partA.h5')
x_f=[]
for i in range(0,X.shape[0]):
    x_f.append(X[i].flatten())

x_f=np.array(x_f)


for i in range(0,Y.shape[0]):
    if(Y[i]==7):
        Y[i]=0
    else:
        Y[i]=1

y_f=[]
for i in range(0,Y.shape[0]):
    temp = np.zeros(2)
    temp[Y[i]]=1
    y_f.append(temp)
y_f=np.array(y_f)
'''
acr=[]
nn=NN((100,50),0.01)
# nn.fit(x_f[:10000],y_f[:10000],4)
#
# nn.score(x_f[10000:],y_f[10000:])
for epochs in [10]:

    ( k_c_validate(k_f=5,training_model=nn,traindata_y=y_f,traindata_x=x_f,epoch=epochs))
# print acr
# #
# # epoch = [2,4,6,8,10]
# #
#
# # acr=[]
# # for i in epoch
# #     nn.fit(x=x_f,y=y_f,i)
# #     acr.append(nn.score(x=x_t,y=y_t))
# #
# # epoch_mat = np.array(epoch)
# # acr_mat = np.array(acr)
# #
# # plt.plot(epoch_mat,acr_mat)
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epochs')
# # plt.title('Softmax-Sigmoid NN')
# # plt.show()
# nn = NN((100,50),0.006)
# nn.fit(x=x_f,y=y_f,it=4)
# print nn.score(x_t,y_t)
# joblib.dump(nn,'Model1/relu1.sav')
'''
nn=joblib.load('Model1/relu_test.sav')
print nn.score(x_f,y_f)

