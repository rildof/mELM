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
        self.NumberofClasses = len(np.unique(conjuntoTreinamento[:,0]))
        self.conjuntoTreinamento = conjuntoTreinamento #Dataset
        self.conjuntoTreinemantoELM = copy.deepcopy(self.conjuntoTreinamento)
        self.feature_saturation = np.zeros((self.NumberofClasses,
                                       self.NumberofInputNeurons))
        self.InputWeight = np.zeros((self.NumberofClasses,
                                self.NumberofInputNeurons))
        self.InputWeightClass = np.zeros((self.NumberofClasses))
    
        
    def get_xai_weights(self):

        
        level = 0
        count_level = 1

        while True:
            level += 1

            self.xai_weights_aux_func()

            if level == 1:
                #Remove the zeroes from the InputWeight matrix
                InputWeight = InputWeight[~np.all(InputWeight == 0, axis=1)]
                InputWeightClass = InputWeightClass[InputWeightClass != 0]
            
            self.run_elm_level()
            #Condições de parada

            if self.conjuntoTreinemanto.size == 0:
                break
            if len(set(self.conjuntoTreinamento[:, 0])) == 1 and ({1} in [set(f) for f in feature_saturation]): # 
                break
            # Verifica se todas as features estão saturadas
            if np.all(feature_saturation == 1):
                print('All features are saturated')
                break
        return self.weights
       


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


