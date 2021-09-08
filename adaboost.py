
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D=np.ones(len(y)) / len(y)
        for i in range(self.T):
            self.h[i]=self.WL(D,X,y)
            y_est=self.h[i].predict(X)
            eps_t= np.sum(D*np.abs(y_est-y)/2)
            self.w[i]=0.5*np.log(-1+1/eps_t)
            D *= np.exp((-1) * self.w[i] * y * y_est)
            D=D/np.sum(D)
        return D
    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        y_est=np.zeros(X.shape[0])
        for i in range(max_t):
            y_est+=self.w[i]*self.h[i].predict(X)
        return np.sign(y_est)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        return sum(abs(y - y_hat) / 2) / y.shape[0]

