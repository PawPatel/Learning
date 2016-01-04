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
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')
    trainset, testset = splitdataset(dataset)    
    return trainset,testset

def loadmadelon():
    datasettrain = np.genfromtxt('madelon/madelon_train.data', delimiter=' ')
    trainlab = np.genfromtxt('madelon/madelon_train.labels', delimiter=' ')
    trainlab[trainlab==-1] = 0
    trainsetx = np.hstack((datasettrain, np.ones((datasettrain.shape[0],1))))
    trainset = (trainsetx, trainlab)
    
    datasettest = np.genfromtxt('madelon/madelon_valid.data', delimiter=' ')
    testlab = np.genfromtxt('madelon/madelon_valid.labels', delimiter=' ')
    testlab[testlab==-1] = 0
    testsetx = np.hstack((datasettest, np.ones((datasettest.shape[0],1))))
    testset = (testsetx, testlab)
      
    return trainset,testset

if __name__ == '__main__':
    #trainset, testset = loadsusy()
    trainset, testset = loadmadelon()
    print('Running on train={0} and test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
    nnparams = {'ni': trainset[0].shape[1], 'nh': 64, 'no': 1}
    classalgs = {'Random': algs.Classifier(),
                 'Linear Regression': algs.LinearRegressionClass(),
                 #'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 #'Naive Bayes Ones': algs.NaiveBayes(),
                 'Logistic Regression': algs.LogitReg(),
                 'Logistic Regression with L1 Reg': algs.LogitRegL1(),
                 'Logistic Regression with L2 Reg': algs.LogitRegL2(),
                 'Logistic Regression with L3 Reg': algs.LogitRegL3()
                 #'Compare Logistic Regression' : algs.compareLogitReg(),
                 #'Neural Network': algs.NeuralNet(nnparams)
                 }
                          
    for learnername, learner in classalgs.iteritems():
        print 'Running learner = ' + learnername
        # Train model
        learner.learn(trainset[0], trainset[1])
        # Test model
        predictions = learner.predict(testset[0])
        accuracy = getaccuracy(testset[1], predictions)
        print 'Accuracy for ' + learnername + ': ' + str(accuracy)
 
