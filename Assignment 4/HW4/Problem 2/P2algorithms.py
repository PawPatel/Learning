import numpy as np
import utilities as utils
from sklearn import cluster

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params=None ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features = range(10)  
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class RidgeRegression(Regressor):
    """
    Ridge Regression with feature selection
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features = range(10)  # max 230 features
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        lam = 1000
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless) + lam*np.identity(Xless.shape[1])), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest
    
class NormalKernel(Regressor):
    #Normal Kernel with regularizer

    def __init__( self, params=None ):
        self.weights = None
        self.features = range(230)  # max 230 features
        self.centers = None 
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        lam = 100

        self.centers = cluster.k_means(Xless, 10)[0]
        Xless = np.dot(Xless, self.centers.T)

        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless) + lam*np.identity(Xless.shape[1])), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features] 
        Xless = np.dot(Xless, self.centers.T)       
        ytest = np.dot(Xless, self.weights)      
        return ytest

class PolyKernel(Regressor):
    # Polynomial Kernel with regularizer

    def __init__( self, params=None ):
        self.weights = None
        self.features = range(230)  # max 230 features
        self.d = 2
        self.centers = None 

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        lam = 100 # set lambda for regularizer coefficient

        self.centers = cluster.k_means(Xless, 10)[0]
        Xless = (1 + np.dot(Xless, self.centers.T))**self.d


        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless) + lam*np.identity(Xless.shape[1])), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]  
        Xless = (1 + np.dot(Xless, self.centers.T))**self.d
        ytest = np.dot(Xless, self.weights)       
        return ytest

class RadialKernel(Regressor):
    # Radial Kernel with regularizer

    def __init__( self, params=None ):
        self.weights = None
        self.features = range(230)  # max 230 features
        self.centers = None
        self.s = 2
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        lam = 1000 # set regularizer coeff

        num = Xless.shape[0]
        XKer = np.zeros((num, 10))
        
        self.centers = cluster.k_means(Xless, 10)[0]

        #transform Data
        for i in range(num):
            for j in range(10):
                XKer[i, j] = np.exp(-(np.linalg.norm(Xless[i, ] - self.centers[j]))/2*self.s**2)


        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(XKer.T,XKer) + lam*np.identity(XKer.shape[1])), XKer.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]  
        #transform Data
        Xker = np.zeros((Xless.shape[0], 10))
        for i in range(Xless.shape[0]):
            for j in range(10):
                Xker[i, j] = np.exp(-(np.linalg.norm(Xless[i, ] - self.centers[j]))/2*self.s**2)

        ytest = np.dot(Xker, self.weights)       
        return ytest
    