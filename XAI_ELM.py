import os 
import numpy as np
from itertools import product
from melm_lib import *
from scipy.linalg import pinv
from scipy.stats import mode
import time
import copy
from sklearn.datasets import make_classification
import re
import numpy as np
import random
from XAI_IterativeWeights import IterativeWeights
from XAI_Plotter import Plotter

class XAI:
    def __init__(self, dataSet, T, P, TVP):
        self.dataSet, self.T, self.P, self.TVP = dataSet, T, P, TVP
        self.NumberofInputNeurons = dataSet[:,1:].shape[1] #Number of features
        self.NumberofTrainingData = dataSet.shape[0] #Number of samples
        pass

    def run_xai_elm(self, ):
        """Function to run the XAI algorithm with the given parameters."""



    def run_traditional_elm(self, InputWeight, BiasofHiddenNeurons,
                            ActivationFunction='dilation',
                            verbose=False):
        """Function to run the traditional ELM algorithm with the given parameters."""
        # Calculate the hidden layer output matrix (H)

        H = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, self.P)
        
        # Calculate the output weights using the pseudoinverse
        OutputWeight = np.linalg.pinv(H.T) @ self.T.T
        Y = (H.T @ OutputWeight).T
        del H  # Clear H to save memory

        # Evaluate the training network
        (acc,
         wrongIndexes) = self.evaluate_network_accuracy(Y, self.T)
        if verbose: print(f'Training Accuracy: {acc}%')
        # Calculate the hidden layer output matrix for the test data (H_test)
        H_test = switchActivationFunction(ActivationFunction, 
                                          InputWeight, BiasofHiddenNeurons, self.TVP)
        TY = (H_test.T @ OutputWeight).T
        del H_test  # Clear H_test to save memory
        (acc,
         wrongIndexes) = self.evaluate_network_accuracy(TY, self.T)
        if verbose: print(f'Testing Accuracy: {acc}%')
        # Evaluate the network for testing
        xx1, yy1, xx2, yy2 = self.separate_classes_plotting(TY, self.T)
        return (xx1, yy1, xx2, yy2)
    
    def evaluate_network_accuracy(self, Y, T):
        """Function to evaluate the network."""
        NumberofTrainingData = Y.shape[1]
        # Get the index of the maximum output (winner neuron) for each pattern
        nodoVencedorRede = np.argmax(Y, axis=0)
        nodoVencedorDesejado = np.argmax(T, axis=0)

        # Count the number of misclassifications
        classificacoesErradas = np.sum(nodoVencedorRede != nodoVencedorDesejado)
        wrongIndexes = np.where(nodoVencedorRede != nodoVencedorDesejado)
        # Calculate accuracy
        accuracy = 1 - (classificacoesErradas / NumberofTrainingData)
        accuracy = round(accuracy * 100, 2)
        
        return accuracy, wrongIndexes

    def separate_classes_plotting(self, TY, TVP):
        """Function to separate the classes for plotting."""
        # Encontra o valor máximo e o índice do vencedor
        nodoVencedorRede = np.argmax(TY, axis=0)
        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for padrao in range(TVP.shape[1]):
            if nodoVencedorRede[padrao] == 0: 
                x1.append(TVP[0, padrao])
                y1.append(TVP[1, padrao])
            else:
                x2.append(TVP[0, padrao])
                y2.append(TVP[1, padrao])

        # Converte as listas em arrays do numpy (opcional, dependendo do uso posterior)
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)

        return x1, y1, x2, y2