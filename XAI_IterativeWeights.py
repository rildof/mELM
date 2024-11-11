import os 
import numpy as np
from itertools import product
from melm_lib import *
from scipy.linalg import pinv
from scipy.stats import mode
import time
import copy
import re
import numpy as np
import random
class IterativeWeights:
    def __init__(self, conjuntoTreinamento, max_iterations=50):
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
        self.InputWeightSaturation = np.zeros((self.NumberofClasses,
                                            self.NumberofInputNeurons))
        self.iteration = 0
        self.count_iteration = 1
        self.max_per_class = self.calculate_max_per_class()


    def get_xai_weights(self):
        """Function that calculates the weights for the XAI algorithm"""
        # Loop through the levels of the XAI algorithm
        # Each level results in num_classes weights added to weights list
        
        while self.iteration < self.max_iterations:
            self.iteration += 1

            self.xai_weights_aux_func()

            if self.iteration == 1:
                #Remove the zeroes from the InputWeight matrix
                self.InputWeight = self.InputWeight[~np.all(self.InputWeight == 0, axis=1)]
                self.InputWeightClass = self.InputWeightClass[self.InputWeightClass != 0]
            
            self.run_elm_level()

            breakpoint()
            #Condições de parada

            if self.conjuntoTreinamento.size == 0:
                break
            if len(set(self.conjuntoTreinamento[:, 0])) == 1 and ({1} in [set(f) for f in self.feature_saturation]): # 
                break
            # Verifica se todas as features estão saturadas
            if np.all(self.feature_saturation == 1):
                print('All features are saturated')
                break
        return self.InputWeight


    def xai_weights_aux_func(self):
        """Function that calculates the weights for the XAI algorithm
        for its current iteration
        
        Consists in three main steps:
        1. Calculate the mode for each class and feature
        2. Calculate the weights for each class and feature
        3. Calculate the saturation for each class and feature"""

        separated_classes = [self.conjuntoTreinamento[self.conjuntoTreinamento[:, 0] == i] for
                              i in range(1, self.NumberofClasses + 1)]
        #remove zero length classes
        separated_classes = [x for x in separated_classes if x.size != 0]
        breakpoint()
        #Min for each class and feature
        min_classes = np.zeros((self.NumberofClasses, self.NumberofInputNeurons))
        for i in range(len(separated_classes)):
            for j in range(self.NumberofInputNeurons):
                # Calculate the min for each class and feature
                min_classes[i, j] = np.min(separated_classes[i][:, j + 1])
        #sort classes by length
        separated_classes = sorted(separated_classes, key=lambda x: len(x))
        #INICIO PESOS_XAI_CLASSE_POR_CLASSE
        for c in separated_classes:
            classe = int(c[0,0])
            #Se saturou para determinada classe
            if sum(self.feature_saturation[int(c[0,0])-1]) == self.NumberofInputNeurons:
                #INICIO insere_amostra_apos_saturacao
                for ii in range(c.shape[0]):
                    # Get new sample (excluding class label)
                    nova_amostra = c[ii, 1:]
                    
                    # Check if sample exists in InputWeight matrix
                    amostra_existente = False
                    if self.InputWeight.size > 0:  # Only check if matrix is not empty
                        # Compare with each existing row
                        for row in self.InputWeight:
                            if np.array_equal(row, nova_amostra):
                                amostra_existente = True
                                break
                    # If sample doesn't exist, add it
                    if not amostra_existente:
                        # Add new sample to weights matrix
                        self.InputWeight = np.vstack((self.InputWeight, nova_amostra))
                        # Set class for this weight
                        self.InputWeightClass[self.count_iteration] = classe
                        # Set saturation status
                        self.InputWeightSaturation[self.count_iteration, :] = self.feature_saturation[classe-1, :]
                        # Increment counter
                        self.count_iteration += 1
                #FIM insere_amostra_apos_saturacao
                continue

            # Set new line for matrices
            self.InputWeight = np.vstack((self.InputWeight, 
                                          np.zeros((1, self.NumberofInputNeurons))))
            self.InputWeightClass = np.append(self.InputWeightClass, 0)
            
            #Calculate the mode for each class and feature (pesos_xai_classe_por_classe)

            # Calculate the mode for each class and feature
            
            # Verifica se a saturação para a classe já ocorreu
            if (np.sum(self.feature_saturation[classe-1, :]) == 
                self.feature_saturation.shape[1]):
                #insere_amostra_apos_saturacao
                # Process each sample in current class data
                for sample in c:
                    new_sample = sample[1:]  # Exclude class label
                    # Check if sample exists using array comparison
                    sample_exists = np.any(np.all(self.InputWeight == new_sample, axis=1))
                    
                    if not sample_exists:
                        # Add new sample to weights
                        self.InputWeight[-1] = new_sample
                        self.InputWeightClass[-1] = sample[0]
                        self.feature_saturation[classe-1, :] = 1  # Set saturation for class
                        self.count_iteration += 1
                continue
            
            def calcula_modas(c, classe):
                # Initialize arrays
                #modas = np.zeros((c.shape[0], self.NumberofInputNeurons, 2))
                modas = []
                # Calculate modes for each feature
                for ii in range(self.NumberofInputNeurons):
                    # Get unique values and their frequencies
                    unique_vals, counts = np.unique(c[:, ii+1], return_counts=True)
                    
                    # Combine frequencies and values
                    modas_frequencias = np.column_stack((counts, unique_vals))
                    
                    # Sort based on class and frequency
                    if self.max_per_class[ii] == classe:
                        # Ordena os valores em ordem decrescente
                        modas_ordenadas = modas_frequencias[modas_frequencias[:,1].argsort()[::-1]]
                    else:
                        # Ordena pela frequência em ordem decrescente
                        modas_ordenadas = modas_frequencias[modas_frequencias[:,0].argsort()[::-1]]
                    
                    # Store sorted modes and frequencies
                    n_rows = modas_ordenadas.shape[0]
                    modas.append(modas_ordenadas)
                    #modas[:n_rows, ii, 1] = modas_ordenadas[:, 0]  # Frequencies
                    #modas[:n_rows, ii, 0] = modas_ordenadas[:, 1]  # Values
                # Calculate saturation
                vector_modas_saturation = []
                for index, features in enumerate(modas):
                    vector_modas_saturation.append(np.zeros(len(features)))
                    # Check if value exists in InputWeight
                    for j, moda in enumerate(features):
                        if np.any(np.all(self.InputWeight == moda[1], axis=1)):
                            vector_modas_saturation[index][j] = 1
                        else:
                            vector_modas_saturation[index][j] = 0
                return modas, vector_modas_saturation
                #FIM calcula_modas
            
            (modas, 
             vector_modas_saturation) = calcula_modas(
                 c, classe)
            # INICIO CALCULA_MAXIMOS
            def calcula_maximos_auxiliar(c, modas, vector_modas_saturation):
                """Calculate auxiliary maximums for XAI weights calculation"""
                vectorInput = np.zeros(self.NumberofInputNeurons)
                vectorFreq = np.zeros(self.NumberofInputNeurons)
                
                flag = 0
                for ii in range(self.NumberofInputNeurons):
                    flag = 0
                    for index, moda in enumerate(modas[ii]):
                        if vector_modas_saturation[ii][index] == 0:
                            vectorInput[ii] = moda[1]
                            vectorFreq[ii] = moda[0]
                            flag = 1
                            break
                    if flag == 0:
                        self.feature_saturation[classe-1][ii] = 1
                if flag == 0:
                    self.feature_saturation[classe-1, :] = 1
                    return vectorInput, False, ii, False
                # Get the mode with the highest frequency

                modeMax, index = np.max(vectorFreq), np.argmax(vectorFreq)
                if modeMax == 1: #Se a maior frequencia for 1
                    # Find max absolute value in vectorInput
                    modeMax = np.max(np.abs(vectorInput))
                    # Find indices where value equals modeMax
                    index = np.where(vectorInput == modeMax)[0]
                    
                    # If no direct match found, check absolute values
                    if len(index) == 0:
                        index = np.where(np.abs(vectorInput) == modeMax)[0]
                        modeMax = -modeMax  # Flip sign if using absolute value match
                    
                    # Take first index if multiple found
                    index = index[0]
                else:
                    # Use value from vectorInput at max frequency position
                    modeMax = vectorInput[index]
                #if modeMax
                
                return vectorInput, modeMax, index, True  # Added flag=True here
            vectorInput, modeMax, index, flag = calcula_maximos_auxiliar(c, modas, vector_modas_saturation)
            
            if  (flag==False):
                inputMax = False
            else:
                # Filtra as linhas onde o valor na coluna index é igual ao inputMax
                linhasSelecionadas = c[c[:, index+1] == modeMax, :].astype(np.int64)[0]
                
                selectedValues = c[linhasSelecionadas, :]
                modas_temp, _ = calcula_modas(selectedValues, classe)
                def calcula_maximos_auxiliar_linhas_selecionadas(c, modas):
                    """Calculate auxiliary maximums for XAI weights calculation for selected lines
                    Simplified version of calcula_maximos_auxiliar without saturation handling
                    """
                    vectorInput = np.zeros(self.NumberofInputNeurons)
                    vectorFreq = np.zeros(self.NumberofInputNeurons)
                    
                    # Get first mode for each feature
                    for ii in range(self.NumberofInputNeurons):
                        if len(modas[ii]) > 0:  # Check if there are modes for this feature
                            vectorInput[ii] = modas[ii][0][1]  # First mode value
                            vectorFreq[ii] = modas[ii][0][0]   # First mode frequency
                    
                    # Find maximum frequency and its index
                    modeMax = np.max(vectorFreq)
                    index = np.argmax(vectorFreq)
                    
                    # Special handling when maximum frequency is 1
                    if modeMax == 1:
                        modeMax = np.max(np.abs(vectorInput))
                        index = np.where(vectorInput == modeMax)[0]
                        
                        if len(index) == 0:
                            index = np.where(np.abs(vectorInput) == modeMax)[0]
                            modeMax = -modeMax
                        
                        index = index[0]
                    else:
                        modeMax = vectorInput[index]
                    
                    return vectorInput, modeMax, index
                (vectorInput, 
                inputMax, 
                index) = calcula_maximos_auxiliar_linhas_selecionadas(
                    linhasSelecionadas, modas_temp)
            #FIM calcula_maximos
            ii = self.NumberofInputNeurons
            count_vector = np.zeros(self.NumberofInputNeurons)
            if flag == False:
                #INICIO ESTUDA_SATURACAO
                if np.all(self.feature_saturation[:,:]==1):
                    print('Saturação total.')
                elif(np.sum(self.feature_saturation[classe-1,:]) == self.NumberofInputNeurons):
                    print('Saturação para a classe', classe)
                    #if size(InputWeightTotal,1)>count_nivel
                    if self.InputWeight.shape[0] > self.count_iteration:
                        #Remove all added zeros in InputWeightTotal and InputWeightClass
                        self.InputWeight = self.InputWeight[:-1]
                        self.InputWeightClass = self.InputWeightClass[:-1]
                        self.InputWeightSaturation = self.InputWeightSaturation[:-1]
                #FIM ESTUDA_SATURACAO
            
            #INICIO update_pesos
            self.InputWeight[-1] = vectorInput
            self.InputWeightClass[-1] = classe
            self.InputWeightSaturation[-1, :] = self.feature_saturation[classe-1, :]
            for ii in range(self.NumberofInputNeurons):
                jj = np.where(modas[ii][:, 1] == vectorInput[ii])[0]
                vector_modas_saturation[ii][jj] = 1
            flag = True
            #FIM update_pesos
            
            linha_com_valor_unico = False
            if self.count_iteration > 1:
                for i in range(self.InputWeight.shape[0]):
                    if np.all(self.InputWeight[i, :] == self.InputWeight[i, 0]):
                        linha_com_valor_unico = True
                        linha = i
                        break
                if linha_com_valor_unico:
                    print('erro')
                    print(self.InputWeight)
                    print(self.feature_saturation)
                    print(self.count_iteration)
                    print(classe)
                    breakpoint()
            #FIM pesos_xai_classe_por_classe


    def run_elm_level(self, ActivationFunction='dilation'):
        """Function that runs the ELM algorithm for 
        the current iteration for evaluation in the next
        iteration"""
        #INICIO dados_entrada_xai
        REGRESSION = 0
        CLASSIFIER = 1
        Elm_Type = CLASSIFIER
        saidasTreinamento = self.conjuntoTreinamento[:, 0]
        entradasTreinamento = self.conjuntoTreinamento[:, 1:]

        saidasTeste = self.conjuntoTreinamento[:, 0]
        entradasTeste = self.conjuntoTreinamento[:, 1:]

        if ActivationFunction in ['bitwise_dilation', 'bitwise_erosion']:
            P = entradasTreinamento.astype(np.int32)
            TVP = entradasTeste.astype(np.int32)
        else:
            P = entradasTreinamento.T
            TVP = entradasTeste.T
        
        T = saidasTreinamento
        TVT = saidasTeste
        NumberofTrainingData = P.shape[1]
        NumberofTestingData = TVP.shape[1]

        if Elm_Type != REGRESSION:
            # Preprocessing the data for classification
            sorted_target = np.sort(np.concatenate((T, TVT), axis=0))
            label = np.unique(sorted_target)
            number_class = len(label)
            NumberofOutputNeurons = number_class

            # Processing the targets of training
            temp_T = np.zeros((NumberofOutputNeurons, NumberofTrainingData))
            for i in range(NumberofTrainingData):
                j = np.where(label == T[i])[0][0]
                temp_T[j, i] = 1
            T = temp_T * 2 - 1

            # Processing the targets of testing
            temp_TV_T = np.zeros((NumberofOutputNeurons, NumberofTestingData))
            for i in range(NumberofTestingData):
                j = np.where(label == TVT[i])[0][0]
                temp_TV_T[j, i] = 1
            TVT = temp_TV_T * 2 - 1
        #FIM dados_entrada_xai
        #INICIO elm_autoral_xai

        def elm_autoral_xai(Elm_Type, ActivationFunction,
                            T, P, TVT, TVP, NumberofTrainingData, 
                            NumberofTestingData):
            InputWeight = self.InputWeight
            NumberofHiddenNeurons = self.NumberofInputNeurons
            etapa = self.iteration
            def avaliacaoRedeELM_XAI(numTeste, saidasRede, saidasDesejada, entrada, treino):
                # Calculating classification error for the test set
                # (The classification rule is winner-takes-all, i.e., the output node that generates the highest output value
                # corresponds to the class of the pattern).
                
                maiorSaidaRede = np.max(saidasRede, axis=0)
                nodoVencedorRede = np.argmax(saidasRede, axis=0)
                
                maiorSaidaDesejada = np.max(saidasDesejada, axis=0)
                nodoVencedorDesejado = np.argmax(saidasDesejada, axis=0)

                classificacoesErradas = 0
                for padrao in range(numTeste - 1, -1, -1):
                    if nodoVencedorRede[padrao] != nodoVencedorDesejado[padrao]:
                        classificacoesErradas += 1
                
                accuracy = 1 - (classificacoesErradas / numTeste)
                print("Accuracy:", accuracy)
                print("Treino:", treino)

                if treino == 'treino':
                    if etapa >= 1:
                        mask = nodoVencedorRede == nodoVencedorDesejado
                        saidasDesejada = saidasDesejada[:, ~mask]
                        entrada = entrada[:, ~mask]

                return accuracy, saidasDesejada, entrada

            # Calculate weights & biases
            start_time_train = time.time()

            # Generate input weights and biases of hidden neurons
            BiasMatrix =  np.zeros((NumberofHiddenNeurons, 1))
            H = switchActivationFunction(ActivationFunction, InputWeight, BiasMatrix,  P)
            # Calculate output weights (beta_i)
            OutputWeight = np.linalg.pinv(H.T) @ T.T
            Y = (H.T @ OutputWeight).T
            
            end_time_train = time.time()
            TrainingTime = end_time_train - start_time_train
            
            # Calculate the output of testing input
            start_time_test = time.time()
            H_test = switchActivationFunction(ActivationFunction, InputWeight, BiasMatrix,  TVP)
            TY = (H_test.T @ OutputWeight).T
            
            end_time_test = time.time()
            TestingTime = end_time_test - start_time_test
            
            if Elm_Type == CLASSIFIER:
                # Calculate the accuracy of the network on the training set
                train_accuracy, T, P = avaliacaoRedeELM_XAI(NumberofTrainingData, Y, T, P, 'treino')
                if etapa != 1:
                    # Calculate the accuracy of the network on the test set
                    test_accuracy, TVT, TVP = avaliacaoRedeELM_XAI(NumberofTestingData, TY, TVT, TVP, etapa, 'teste')
                    #TODO: Implement confusao_funcao_elm
                    #confusao_funcao_elm(T, Y, TVT, TY, iteracao, ActivationFunction, 
                    #                    e_index, c_index, g_index, classificador, fold)
                else:
                    test_accuracy = 0.0
            else:
                train_accuracy = None
                test_accuracy = None

            return T, P, train_accuracy, test_accuracy, TrainingTime, TestingTime
        (T,P,train_accuracy,
        test_accuracy,TrainingTime,
        TestingTime) = elm_autoral_xai(Elm_Type,
        ActivationFunction, T, P, TVT, TVP, 
        NumberofTrainingData, NumberofTestingData,)
        breakpoint()
        #TODO corrigir cada coluna T -> cada classe, -1 não é 1 é da classe
        self.conjuntoTreinamento = np.column_stack((T, P))
        #FIM elm_autoral_xai


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

    def calculate_max_per_class(self):
        """Function that calculates the class with the greatest value for each feature"""
        """In the max_per_class array, there is a value for each feature, which is the class with the greatest value for that feature"""
        max_per_class = np.zeros((1, self.NumberofInputNeurons))
        for i in range(self.NumberofInputNeurons):
            max_class_value = np.argmax(self.conjuntoTreinamento[:, i+1])
            max_per_class[0, i] = self.conjuntoTreinamento[max_class_value, 0]
        return max_per_class[0]
            
if __name__ == '__main__':
        # Load benign and malign data
    from XAI_PreProcessing import DataProcessing
    from XAI_ELM import XAI 
    preProcesser = DataProcessing(None, None)
    dataset, T, P, TVP = preProcesser.get_dataset_scikit(100,10,3,42)
        #dataset, T, P, TVP = (
    #preProcesser.get_sample_datasets('linear'))
    #preProcesser.get_dataset_scikit(500,4,4))

    print('Dataset loaded')
    # Calculate Weights

    weight_factory = IterativeWeights(
         conjuntoTreinamento=dataset,
         max_iterations=20)
    weights_elm, bias_elm = weight_factory.get_xai_weights()
    #weights_elm, bias_elm = weight_factory.get_random_weights(NumberofHiddenNeurons=100)

    print('Weights Calculated')
    # Run XAI algorithm

    xai = XAI(dataset, T, P, TVP)
    #xai.run_xai_elm()