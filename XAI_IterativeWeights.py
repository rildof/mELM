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
        self.NumberofInputNeurons = conjuntoTreinamento[:,1:].shape[1] #Number of features
        self.NumberofClasses = len(np.unique(conjuntoTreinamento[:,0]))
        self.conjuntoTreinamento = conjuntoTreinamento #Dataset
        #Variables for the Iterative XAI algorithm
        self.max_iterations = max_iterations #Max iterations for the XAI algorithm
        self.conjuntoTreinemantoELM = copy.deepcopy(self.conjuntoTreinamento)
        self.feature_saturation = np.zeros((self.NumberofClasses,
                                       self.NumberofInputNeurons))
        self.InputWeight = np.zeros((self.NumberofClasses,
                                self.NumberofInputNeurons))
        self.InputWeightClass = np.zeros((self.NumberofClasses))
        self.iteration = 0
        self.count_iteration = 1
        
    def get_xai_weights(self):
        """Function that calculates the weights for the XAI algorithm"""
        # Loop through the levels of the XAI algorithm
        # Each level results in num_classes weights added to weights list
        while True or self.iteration < self.max_iterations:
            self.iteration += 1

            self.xai_weights_aux_func()

            if self.iteration == 1:
                #Remove the zeroes from the InputWeight matrix
                InputWeight = InputWeight[~np.all(InputWeight == 0, axis=1)]
                InputWeightClass = InputWeightClass[InputWeightClass != 0]
            
            self.run_elm_level()

            #Condições de parada

            if self.conjuntoTreinemanto.size == 0:
                break
            if len(set(self.conjuntoTreinamento[:, 0])) == 1 and ({1} in [set(f) for f in self.feature_saturation]): # 
                break
            # Verifica se todas as features estão saturadas
            if np.all(self.feature_saturation == 1):
                print('All features are saturated')
                break
        return self.weights

    def xai_weights_aux_func(self):
        """Function that calculates the weights for the XAI algorithm
        for its current iteration
        
        Consists in three main steps:
        1. Calculate the mode for each class and feature
        2. Calculate the weights for each class and feature
        3. Calculate the saturation for each class and feature"""

        separated_classes = [self.conjuntoTreinamento[self.conjuntoTreinamento[:, 0] == i] for
                              i in range(1, self.NumberofClasses + 1)]
        #Min for each class and feature
        min_classes = np.zeros((self.NumberofClasses, self.NumberofInputNeurons))
        for i in range(self.NumberofClasses):
            for j in range(self.NumberofInputNeurons):
                # Calculate the min for each class and feature
                min_classes[i, j] = np.min(separated_classes[i][:, j + 1])
        #sort classes by length
        separated_classes = sorted(separated_classes, key=lambda x: len(x))
        for c in separated_classes:
            # Set new line for matrices
            self.InputWeight = np.vstack((self.InputWeight, 
                                          np.zeros((1, self.NumberofInputNeurons))))
            self.InputWeightClass = np.append(self.InputWeightClass, 0)
            
            #Calculate the mode for each class and feature (pesos_xai_classe_por_classe)

            # Calculate the mode for each class and feature
            classe = int(c[0,0])
            # Verifica se a saturação para a classe já ocorreu
            if (np.sum(self.feature_saturation[classe-1, :]) == 
                self.feature_saturation.shape[1]):
                continue
            # Calcula os máximos (calcula_maximos)
            #calcula_maximos_auxiliar
            def calcula_maximos_auxiliar(c):
                vectorInput = np.zeros(self.NumberofInputNeurons)
                for i in range(1, self.NumberofInputNeurons):
                    if self.feature_saturation[classe-1, i] == 0:
                        vectorInput[i] = stats.mode((c[:, i]))[0]
                    else:
                        vectorInput[i] = min_classes[classe-1, i]
                
                filtrado = vectorInput[self.feature_saturation[classe - 1, :] != 1]
                
                # Determina o valor máximo do vetor filtrado
                inputMax = np.max(filtrado)
                index = np.where(vectorInput == inputMax)[0][0]
                return vectorInput, inputMax, index
            vectorInput, inputMax, index = calcula_maximos_auxiliar(c)
            # Filtra as linhas onde o valor na coluna index é igual ao inputMax
            #linhasSelecionadas = entrada[entrada[:, index] == inputMax, :]
            linhasSelecionadas = c[c[:, index] == inputMax, :]
            vectorInput, inputMax, index = calcula_maximos_auxiliar(linhasSelecionadas)
            #FIM calcula_maximos

            


            #FIM pesos_xai_classe_por_classe

            #Condicional para checar se o peso foi populado
            if InputWeightClass[-1] == 0 and self.iteration > 1:
                #Remove all added zeros in InputWeightTotal and InputWeightClass
                InputWeightTotal = InputWeightTotal[:-1]
                InputWeightClass = InputWeightClass[:-1]
            breakpoint()


    def run_elm_level(self):
        """Function that runs the ELM algorithm for 
        the current iteration for evaluation in the next
        iteration"""

        pass


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



if __name__ == '__main__':
        # Load benign and malign data
    from XAI_PreProcessing import DataProcessing
    from XAI_ELM import XAI
    benign_path, malign_path = None, None
    preProcesser = DataProcessing(benign_path, malign_path)
    if benign_path != None and malign_path != None:
         dataset, T, P, TVP = preProcesser.create_dataset()
    else:
         dataset, T, P, TVP = (
        preProcesser.get_sample_datasets('linear'))
        #preProcesser.get_dataset_scikit(500,4,4))

    print('Dataset loaded')
    # Calculate Weights

    weight_factory = IterativeWeights(
         conjuntoTreinamento=dataset,
         max_iterations=1000)
    weights_elm, bias_elm = weight_factory.get_xai_weights()
    #weights_elm, bias_elm = weight_factory.get_random_weights(NumberofHiddenNeurons=100)

    print('Weights Calculated')
    # Run XAI algorithm

    xai = XAI(dataset, T, P, TVP)
    #xai.run_xai_elm()