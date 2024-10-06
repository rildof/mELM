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

class IterativeWeights:
    def __init__(self, conjuntoTreinamento, max_iterations=1000):
        self.max_iterations = max_iterations #Max iterations for the XAI algorithm
        self.NumberofInputNeurons = conjuntoTreinamento[:,1:].shape[1] #Number of features
        self.conjuntoTreinamento = conjuntoTreinamento #Dataset
        self.weights = []
        self.biases = []

    
        
    def get_xai_weights(self):
        conjuntoTreinamento = self.conjuntoTreinamento
       


    def get_random_weights(self, NumberofHiddenNeurons):
        def normalizar_pesos(matriz):
            vetor = matriz.flatten()
            maximo = np.max(vetor)
            minimo = np.min(vetor)

            ra = 0.8
            rb = 0.2

            R = (((ra - rb) * (matriz - minimo)) / (maximo - minimo)) + rb
            
            return R

        # Set the seed for reproducibility
        np.random.seed(random.randint(0, 1000))
        
        # Generate random weights between -1 and 1
        InputWeight = np.random.rand(NumberofHiddenNeurons, self.NumberofInputNeurons) * 2 - 1
        
        # Generate random biases
        np.random.seed(random.randint(0, 1000))
        BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons, 1)
        
        # Normalize weights and biases
        InputWeight = normalizar_pesos(InputWeight)
        BiasofHiddenNeurons = normalizar_pesos(BiasofHiddenNeurons)
        
        return InputWeight, BiasofHiddenNeurons


