from __future__ import division # floating point division
import csv
import random
from math import exp
import numpy as np
from sklearn.decomposition import PCA

import algorithms as algs
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

# Split a dataset into 10 splits, returned through a yield
def dataIn10(dataset):
    numinputs = dataset.shape[1]
    numsplits = 10

    splits = np.array(np.split(dataset, numsplits))

    print splits.shape

    for index in range(numsplits):

        Xtest = splits[index][:, 0:splits[index].shape[1]-1]
        ytest = splits[index][:, splits[index].shape[1]-1]

        mask = np.ones(numsplits, dtype = bool)
        mask[index] = 0
        traindata = np.concatenate(splits[mask])

        Xtrain = traindata[:, 0:traindata.shape[1]-1]
        ytrain = traindata[:, traindata.shape[1]-1]

        yield ((Xtrain, ytrain), (Xtest, ytest))


# Run through all the regressors, using 10 splits of your data, outputing average error
def usesplits(dataset):
    classalgs = {'Random': algs.Regressor(),
                 'Mean': algs.MeanPredictor(),
                 'FSLinearRegression': algs.FSLinearRegression(),
                 'RidgeRegression': algs.RidgeRegression()
                 }


    # Run all through all the algorithms, averaging errors over splits
    error = [0,0,0,0]
    i = 0

    for trainset, testset in dataIn10(dataset):

        for learnername, learner in classalgs.iteritems():
            learner.learn(trainset[0], trainset[1])
            predictions = learner.predict(testset[0])
            error[i] += geterror(testset[1], predictions)
            i += 1

        i = 0

    for learnername, learner in classalgs.iteritems():    
        print 'Accurancy for ' + learnername + ': ' + str(error[i]/10) 
        i +=1 

# Use the original implementation
def useoriginal(dataset):
    trainset, testset = splitdataset(dataset)
    print('Split {0} rows into train={1} and test={2} rows').format(
        len(dataset), trainset[0].shape[0], testset[0].shape[0])
    classalgs = {'Random': algs.Regressor(),
                 'Mean': algs.MeanPredictor(),
                 'FSLinearRegression': algs.FSLinearRegression()
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

# Use PCA to subselect 10 features. Then, usesplits on the new data.
def usePCA(dataset):
    
    # grab the y values
    y = dataset[:, dataset.shape[1]-1, None]

    # do a principal component analysis of the data, using 10 components
    pca = PCA(n_components = 10)
    newdata = pca.fit_transform(dataset)
    # throw the y values back on    
    newdata = np.hstack((newdata, y))

    # run the usual algorithms on 10 splits of the data, now represented using the first
    # 10 principal components. This will do very well.
    usesplits(newdata)


# get error for the predictions
def geterror(predictions, ytest):
    # Can change this to other error values
    return utils.l2err_squared(predictions,ytest)/ytest.shape[0]
 

def useSGD(dataset):
    #set the number of features to use
    numfeatures = 10

    # get the dataset using the original implementation. Randomly choose 4500 of the 
    # 5000 datapoints as the training set and the remaining 500 as the test set
    trainset, testset = splitdataset(dataset)

    # Set the data appropriately
    X = trainset[0]
    Xtrain = X[: ,0: numfeatures]
    X2 = testset[0]
    Xtest = X2[:, 0: numfeatures]
    ytrain = trainset[1]
    ytest = testset[1]

    passes = 50 # Set the number of passes through all points in data. 
    alpha = .0000001 # Set the learning rate. A very small rate seems to work best here
    w = np.random.normal(0, .001, 10) # Initialize w. Small values near 0 seem to work well

    for number in range(passes): 
        for i in range(4500):

            w -= alpha*(np.dot(Xtrain[i,:], w) - ytrain[i])* Xtrain[i,:] 
            # error = geterror(ytrain, np.dot(Xtrain, w)) # To see the error decreasing
        
    print 'Stoch. Grad. Descent: ' + str(geterror(ytest, np.dot(Xtest,w)))
    #One should get a reasonable error of around 600. Some runs can get as low as 300, better
    #than linear regression.

def usePoisson(dataset):
    numfeatures = 10
    usepca = True

    trainset, testset = splitdataset(dataset)
    X2 = testset[0]
    Xtest = X2[:, 0: numfeatures]
    ytest = testset[1]
    ytrain = trainset[1]

    if usepca == False:
        # Set the data appropriately
        X = trainset[0]
        Xtrain = X[: ,0: numfeatures]    
    else :
        # do a principal component analysis of the data, using 10 components
        pca = PCA(n_components = numfeatures)
        newdata = pca.fit_transform(np.vstack((trainset[0], testset[0])))
        Xtrain = newdata[0:4500, :]
        Xtest = newdata[4500:, :]

    w = np.random.normal(0, .0000000001, numfeatures)
    c = np.zeros(4500)

    Xtrain = Xtrain/10000
    ytrain = ytrain/10000
    Xtest = Xtest/10000
    ytest = ytest/10000

    for t in range(200):

        for i in range(4500):
            c[i] = exp(np.dot(w, Xtrain[i, :]))
    
        w += np.dot(np.dot( np.linalg.inv(np.dot(np.dot(Xtrain.T, np.diag(c)), Xtrain) + .00001*np.identity(numfeatures)), Xtrain.T), (ytrain - c))

        print 10000*geterror(ytrain, np.dot(Xtrain, w))

    print 'Poisson Regression: ' + str(10000*geterror(ytest, np.dot(Xtest,w)))



if __name__ == '__main__':
    # Prepare dataset
    filename = 'blogData_train_small.csv'
    dataset = utils.loadcsv(filename)   
    numinputs = dataset.shape[1]

    option = 'poisson' #set this to run the appropriate algorithm
    # splits = run random, mean, linreg, ridgereg on 10 splits of the data
    # pca = subselect 10 features using first 10 principal comp, then splits on new data
    # sgd = run stoch grad descent for linear regression. 
    # poisson = run Poisson regression. inside usePoisson() there are more options
    # else, run the original given implementation

    if option == 'splits':
        usesplits(dataset[:, 50:])
    elif option == 'pca':
        usePCA(dataset[:, 50:])
    elif option == 'sgd':
        useSGD(dataset[:, 50:])
    elif option == 'poisson':
        usePoisson(dataset[:,50:])
    else:
        useoriginal(dataset[:, 50:])


