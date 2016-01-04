import csv
import random
import math
import numpy as np
from scipy.stats import norm

import classalgorithms as algs
 
def splitdataset(dataset, trainsize=500, testsize=300, testfile=None):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]        
        
    # Add a column of ones; done after to avoid modifying entire dataset
    #Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    #Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    

    return ((Xtrain,ytrain), (Xtest,ytest))

# Split a dataset into 10 splits, returned through a yield
def dataIn10(dataset):
    numinputs = dataset.shape[1]
    numsplits = 10

    splits = np.array(np.split(dataset, numsplits))

    for index in range(numsplits):

        Xtest = splits[index][:, 0:splits[index].shape[1]-1]
        ytest = splits[index][:, splits[index].shape[1]-1]

        mask = np.ones(numsplits, dtype = bool)
        mask[index] = 0
        traindata = np.concatenate(splits[mask])

        Xtrain = traindata[:, 0:traindata.shape[1]-1]
        ytrain = traindata[:, traindata.shape[1]-1]

        yield ((Xtrain, ytrain), (Xtest, ytest))

 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
    return np.genfromtxt('susysubset.csv', delimiter=',')  
    

#MAIN FUNCTION
if __name__ == '__main__':
    learnerLinReg = algs.LinearRegressionClass()
    learnerLogReg = algs.LogitReg()

    paramLinReg = np.array([.01, .05, .10, .50, 1, 2, 5])
    paramLogRegIter = np.array([5, 10, 20, 50, 100, 500])
    paramLogRegStepsize = np.array([.01, .05, .10, .50, 1, 2, 5])

    errorLinReg = np.zeros(10)
    errorLogReg = np.zeros(10)

    dataset = loadsusy()
    print dataset.shape

    i=0 
    for trainset, testset in dataIn10(dataset):
        
        maxj = 0
        maxk = 0
        maxl = 0
        acc = 0

        # Iterate through all params for LinReg, indexed by j, and store average error for each in avgerrorLinReg[j]
        for j in range(paramLinReg.size):
            
            avgacc = 0

            for trainfold, testfold in dataIn10(np.column_stack((trainset[0], trainset[1]))):
                learnerLinReg.learn(trainfold[0], trainfold[1], paramLinReg[j])
                predictions = learnerLinReg.predict(testfold[0])
                avgacc += getaccuracy(testfold[1], predictions)/10
            if avgacc > acc: 
                acc = avgacc
                maxj = j
            

        print acc

        acc = 0
        # Iterate through all params for LogReg, indexed by k,l, and store avg error in avgerrorLogReg(k,l)
        for k in range(paramLogRegIter.size):
            for l in range(paramLogRegStepsize.size):

                avgacc = 0
                for trainfold, testfold in dataIn10(np.column_stack((trainset[0], trainset[1]))):
                    learnerLogReg.learn(trainfold[0], trainfold[1], paramLogRegStepsize[l], paramLogRegIter[k])
                    predictions = learnerLogReg.predict(testfold[0])
                    avgacc += getaccuracy(testfold[1], predictions)/10
                if avgacc > acc:
                    acc = avgacc
                    maxk = k
                    maxl = l
                   
        print acc    

        learnerLinReg.learn(trainset[0], trainset[1], paramLinReg[maxk])
        predictions = learnerLinReg.predict(testset[0])
        errorLinReg[i] = getaccuracy(testset[1], predictions)


        learnerLogReg.learn(trainset[0], trainset[1], paramLogRegStepsize[maxl], paramLogRegIter[maxk])
        predictions = learnerLogReg.predict(testset[0])
        errorLogReg[i] = getaccuracy(testset[1], predictions)

        i+=1

    print "Errors from Linear Regression: " + errorLinReg
    print "Errors from Logistic Regression: " + errorLogReg
    

    print 'Finished!'

