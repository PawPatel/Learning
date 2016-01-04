from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from scipy.stats import norm

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
        
        self.xmeans = None
        self.cstd = None
        self.py = None

    def learn(self, Xtrain, ytrain):
        # Find P(Y=0) and P(Y=1) and set in array
        self.py = np.array([1 - sum(ytrain)/ytrain.size, sum(ytrain)/ytrain.size])

        y1Index = np.nonzero(ytrain)
        y0Index = np.nonzero(1-ytrain)

        self.xmeans = np.empty((Xtrain.shape[1], 2))
        self.xstd = np.empty((Xtrain.shape[1], 2))

        for i in range(Xtrain.shape[1]):
            self.xmeans[i, 0] = np.mean(Xtrain[y0Index, i])
            self.xmeans[i, 1] = np.mean(Xtrain[y1Index, i])
            self.xstd[i, 0] = np.std(Xtrain[y0Index, i])
            self.xstd[i, 1] = np.std(Xtrain[y1Index, i])
       
    def predict(self, Xtest):
        pred = np.zeros(Xtest.shape[0])
        if self.usecolumnones == False: 
            adjust = -1
        else:
            adjust = 0

        for i in range(Xtest.shape[0]):
            yis0 = self.py[0]
            yis1 = self.py[1]

            for j in range(Xtest.shape[1] + adjust):
                yis0 *= norm(self.xmeans[j,0], self.xstd[j,0]).pdf(Xtest[i,j])
                yis1 *= norm(self.xmeans[j,1], self.xstd[j,1]).pdf(Xtest[i,j])

            if(yis0 < yis1):
                pred[i] = 1
            else:
                pred[i] = 0

        return pred 


    
class LogitReg(Classifier):

    def __init__( self, params=None ):
        self.weights = None
        self.stepsize = 1
    
    def learn(self, Xtrain, ytrain):
        iterate = 10 #set num of iterations to perform descent
        I = np.identity(Xtrain.shape[0])

        #initialize the weights using linear regression
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)

        for i in range(iterate):
            p = utils.sigmoid(np.dot(Xtrain, self.weights))
            P = np.diag(p)
            # Use Hessian or Not (for Madelon data set, Hessian is expensive to compute) First line is Newton, Second is Grad Desc
            self.weights += np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Xtrain.T, P), np.dot(I-P, Xtrain))), Xtrain.T), ytrain - p)  
            # self.weights +=  -self.stepsize*np.dot(Xtrain.T, ytrain - p) 

    def predict(self, Xtest):
        pred = utils.sigmoid(np.dot(Xtest, self.weights))
        pred[pred < .5] = 0
        pred[pred > .5] = 1

        return pred

class LogitRegL1(Classifier):

    def __init__( self, params=None ):
        self.weights = None
        self.stepsize = 1
        self.alpha = 1
    
    def learn(self, Xtrain, ytrain):
        iterate = 1000 #set num of iterations to perform descent
        I = np.identity(Xtrain.shape[0])

        #initialize the weights using linear regression
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)

        for i in range(iterate):
            p = utils.sigmoid(np.dot(Xtrain, self.weights))
            P = np.diag(p)
            # self.weights += np.dot(np.linalg.inv(np.dot(np.dot(Xtrain.T, P), np.dot(I-P, Xtrain))), np.dot(Xtrain.T, ytrain - p) + self.alpha*(self.weights/np.absolute(self.weights))) 
            self.weights +=  -self.stepsize*np.dot(Xtrain.T, ytrain - p) + self.alpha*(self.weights/np.absolute(self.weights)) 

    def predict(self, Xtest):
        pred = utils.sigmoid(np.dot(Xtest, self.weights))
        pred[pred < .5] = 0
        pred[pred > .5] = 1

        return pred

class LogitRegL2(Classifier):

    def __init__( self, params=None ):
        self.weights = None
        self.alpha = 1
    
    def learn(self, Xtrain, ytrain):
        iterate = 10 #set num of iterations to perform descent
        I = np.identity(Xtrain.shape[0])

        #initialize the weights using linear regression
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)

        for i in range(iterate):
            p = utils.sigmoid(np.dot(Xtrain, self.weights))
            P = np.diag(p)
            self.weights += np.dot(np.linalg.inv(np.dot(np.dot(Xtrain.T, P), np.dot(I-P, Xtrain)) + np.identity(self.weights.size)), np.dot(Xtrain.T, ytrain - p) + np.absolute(self.weights))  

    def predict(self, Xtest):
        pred = utils.sigmoid(np.dot(Xtest, self.weights))
        pred[pred < .5] = 0
        pred[pred > .5] = 1

        return pred


class LogitRegL3(Classifier):

    def __init__( self, params=None ):
        self.weights = None
        self.alpha = 3
    
    def learn(self, Xtrain, ytrain):
        iterate = 10 #set num of iterations to perform descent
        I = np.identity(Xtrain.shape[0])

        #initialize the weights using linear regression
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)

        for i in range(iterate):
            p = utils.sigmoid(np.dot(Xtrain, self.weights))
            P = np.diag(p)
            self.weights += np.dot(np.linalg.inv(np.dot(np.dot(Xtrain.T, P), np.dot(I-P, Xtrain)) + self.alpha*np.diag(np.absolute(self.weights))), np.dot(Xtrain.T, ytrain - p) + self.alpha*self.weights**2)  

    def predict(self, Xtest):
        pred = utils.sigmoid(np.dot(Xtest, self.weights))
        pred[pred < .5] = 0
        pred[pred > .5] = 1

        return pred


# Classifier for problem 2, which we compare to logistic regression above.
class compareLogitReg(Classifier):
    def __init__(self, params=None):
        self.weights = None
        self.stepsize = .1

    def learn(self, Xtrain, ytrain):
        iterate = 10 #set num of iterations to perform descent
        I = np.identity(Xtrain.shape[0])

        #initialize the weights using linear regression
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)

        for i in range(iterate):
            wx = np.dot(Xtrain, self.weights)
            A = 1/ (1+ wx/np.sqrt(1+wx**2))
            B = 1/ (1+ wx/np.sqrt(1+wx**2))
            C = (np.sqrt(1+wx**2) - (wx**2)*(1+ wx**2)**(-.5))/ (1+wx**2)

            self.weights += self.stepsize*np.dot(Xtrain.T, (ytrain/A + (ytrain -1)/B)*C)


    def predict(self, Xtest):
        pred = utils.sigmoid(np.dot(Xtest, self.weights))
        pred[pred < .5] = 0
        pred[pred > .5] = 1

        return pred 


class NeuralNet(Classifier):
   
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Set step-size
        self.stepsize = 0.01

        # Number of repetitions over the dataset
        self.reps = 5
        
        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp,:],ytrain[samp])
            
    # Need to implement predict function, since currently inherits the default

    def evaluate(self, inputs):
        """ Including this function to show how predictions are made """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = np.ones(self.nh)
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = np.ones(self.no)
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)

    def update(self, inp, out):
        h = self.transfer(np.dot(self.wi, inp))
        yhat = self.transfer(np.dot(self.wo, h))

        delta = (-out/yhat + (1-out)/(1-yhat))*yhat*(1-yhat)

        self.wo = self.wo - self.stepsize* (delta * h.T)
        self.wi = self.wi - self.stepsize* (delta * (self.wo * h * (1-h)).T * inp.T)    
            
    def predict(self, Xtest):
        pred = self.transfer(np.dot(self.wo, self.transfer(np.dot(self.wi, Xtest.T))))
        pred[pred < .5] = 0
        pred[pred > .5] = 1
        return pred.T