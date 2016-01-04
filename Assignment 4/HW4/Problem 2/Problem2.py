from __future__ import division # floating point division
import csv
import random
from math import exp
import numpy as np
from sklearn.decomposition import PCA

import P2algorithms as algs
import utilities as utils


 # Split your dataset into a trainset and test set, of given sizes. 
def splitdataset(dataset, trainsize=4500, testsize=500):

    # Now randomly split into train and test    
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))

# get error for the predictions
def geterror(predictions, ytest):
    # Can change this to other error values
    return utils.l2err_squared(predictions,ytest)/ytest.shape[0]

# Use the original implementation
def useoriginal(dataset):
    trainset, testset = splitdataset(dataset)

    print('Split {0} rows into train={1} and test={2} rows').format(
        len(dataset), trainset[0].shape[0], testset[0].shape[0])
    classalgs = {'Random': algs.Regressor(),
                 'Mean': algs.MeanPredictor(),
                 'FSLinearRegression': algs.FSLinearRegression(),
                 'Normal Kernel': algs.NormalKernel(),
                 'Polynomial Kernel': algs.PolyKernel(),
                 'Radial Kernal': algs.RadialKernel()
                 }


    # Runs all the algorithms on the data and print out results    
    for learnername, learner in classalgs.iteritems():
        print 'Running learner = ' + learnername
        # Train model
        learner.learn(trainset[0], trainset[1])
        # Test model
        predictions = learner.predict(testset[0])
        #print predictions
        accuracy = geterror(testset[1], predictions)
        print 'Accuracy for ' + learnername + ': ' + str(accuracy)

# Main function
if __name__ == '__main__':
    # Prepare dataset
    filename = 'blogData_train_small.csv'
    dataset = utils.loadcsv(filename)   
    numinputs = dataset.shape[1]

	
    useoriginal(dataset[:, 50:])


