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

class XAI:
    def __init__(self, dataSet):
        self.dataSet = dataSet
        self.NumberofInputNeurons = dataSet.shape[1] #Number of features

        pass

    def run_xai_elm(self, ):
        """Function to run the XAI algorithm with the given parameters."""



    def run_traditional_elm(self, InputWeight, ActivationFunction='dilation'):
        """Function to run the traditional ELM algorithm with the given parameters."""
        # Calculate the hidden layer output matrix (H)
        BiasofHiddenNeurons = []



        H = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P)
        
        # Calculate the output weights using the pseudoinverse
        OutputWeight = np.linalg.pinv(H.T) @ T.T
        Y = (H.T @ OutputWeight).T
        del H  # Clear H to save memory

        # Evaluate the training network
        acc, wrongIndexes = avaliarRedeTreino(Y, NumberofTrainingData, T)

        # Calculate the hidden layer output matrix for the test data (H_test)
        H_test = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, TVP)
        TY = (H_test.T @ OutputWeight).T
        del H_test  # Clear H_test to save memory

        # Evaluate the network for testing
        xx1, yy1, xx2, yy2 = avaliarRede(TY, TVP)
        pass
    
    