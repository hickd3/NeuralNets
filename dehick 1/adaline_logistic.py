'''adaline_logistic.py
CS343: Neural Networks
Project 1: Single Layer Networks
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np

from adaline import Adaline


class AdalineLogistic(Adaline):
    '''ADALINE network that performs logistic regression'''

    def activation(self, net_in):
        '''Sigmoid activation function'''
        return 1 / (1 + np.exp(-net_in))

    def predict(self, features):
        ''' Predicts the class of each input (feature) sample using ADALINE'''
        return np.where(self.activation(self.net_input(features)) >= 0.5, 1, 0)

    def loss(self, y, net_act):
        ''' Computes the cross-entropy loss (for a single training epoch)'''
        # print('act is', net_act)
        # print('Log(act) is', np.log(net_act))
        return np.sum(-y*np.log(net_act) - (1-y)*np.log(1-net_act))
