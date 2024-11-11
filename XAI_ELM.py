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


    def run_xai_elm(self, ActivationFunction='dilation'):
        """Function that runs the ELM algorithm for 
        the current iteration for evaluation in the next
        iteration"""
        #INICIO dados_entrada_xai
        REGRESSION = 0
        CLASSIFIER = 1
        Elm_Type = CLASSIFIER
        saidasTreinamento = self.conjuntoTreinamentoELM[:, 0]
        entradasTreinamento = self.conjuntoTreinamentoELM[:, 1:]

        saidasTeste = self.conjuntoTreinamentoELM[:, 0]
        entradasTeste = self.conjuntoTreinamentoELM[:, 1:]

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
            NumberofHiddenNeurons = self.InputWeight.shape[0]
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
                    test_accuracy, TVT, TVP = avaliacaoRedeELM_XAI(NumberofTestingData, TY, TVT, TVP, 'teste')
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

        xx1, yy1, xx2, yy2 = self.separate_classes_plotting(TY, self.T)
        return (xx1, yy1, xx2, yy2)
        #FIM elm_autoral_xai



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