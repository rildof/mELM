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



    def run_traditional_elm(self, NumberofHiddenNeurons, ActivationFunction='dilation'):
        """Function to run the traditional ELM algorithm with the given parameters."""
        weights = IterativeWeights(self.dataSet, 2)
        pass
    
    