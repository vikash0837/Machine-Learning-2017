import numpy as np
import math


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def variance(numbers):
    avg = mean(numbers)
    var = sum([math.pow(x - avg, 2) for x in numbers]) / (float(len(numbers) - 1))
    return var


class GaussianNB(object):
    def __init__(self):
        self.mean_dev = {}
        self.stdprior = np.array([])

    def fit(self, X, Y):
        class_sep = {}
        for i in range(0, len(X)):
            temp = Y[i]
            # print type(Y[i])
            if (temp not in class_sep):
                class_sep[Y[i]] = []

            class_sep[Y[i]].append(X[i])
            # print X[i]

        
        for c_val, instances in class_sep.iteritems():
            self.mean_dev[c_val] = [(mean(attribute), variance(attribute)) for attribute in zip(*instances)]

        self.stdprior = np.zeros(len(class_sep), dtype=float)
        for i in range(0, len(class_sep)):
            self.stdprior[i] = len(class_sep[i]) / float(len(X))

    def predict(self, X):
        Y = np.zeros(len(X))
        for i in range(0, len(X)):
            result = 0
            maxlikelyhood = float('-inf')
            for j in range(0, len(self.stdprior)):
                log_likelyhood = 0.0000
                for k in range(0, len(X[i])):
                    temp = 0.0
                    if(self.mean_dev[j][k][1]!=0):
                        temp = math.exp(
                            (-1) * ((X[i][k] - self.mean_dev[j][k][0]) ** 2) / float(2 * self.mean_dev[j][k][1])) / float(
                            math.sqrt(2 * math.pi * self.mean_dev[j][k][1]))
                    if(temp>0):
                        log_likelyhood = log_likelyhood + math.log(temp)
                if(self.stdprior[j]>0):
                    log_likelyhood+=math.log(self.stdprior[j])
                #print log_likelyhood
                if (log_likelyhood > maxlikelyhood):
                    maxlikelyhood = log_likelyhood
                    result = j
            Y[i] = result

        return Y

