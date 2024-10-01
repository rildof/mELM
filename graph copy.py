import os 
import numpy as np
from itertools import product
from melm_lib import *
from scipy.linalg import pinv
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import mode
import time
import copy
from sklearn.datasets import make_classification
import re
import numpy as np
import random
points_size = 4
samples_size = 10
text_size = 30
def kernel_matrix_rildo(ActivationFunction, InputWeight, BiasMatrix, P):
    NumberofHiddenNeurons = InputWeight.shape[0]
    NumberofTrainingData = P.shape[1]
    H = np.ones((NumberofHiddenNeurons, NumberofTrainingData))
    NumberofAtributos = InputWeight.shape[1]
    if ActivationFunction.lower() in ['dil_classica', 'dilatacao_classica', 'dilation']:
        
        for i in range(NumberofHiddenNeurons):
            for j in range(NumberofTrainingData):
                result = min(InputWeight[i, 0], P[0, j])
                
                for k in range(1, NumberofAtributos):
                    temp = min(InputWeight[i, k], P[k, j])
                    
                    if temp > result:
                        result = temp
                        
                H[i, j] = result
                
    elif ActivationFunction.lower() in ['ero_classica', 'erosao_classica', 'erosion']:
        
        for i in range(NumberofHiddenNeurons):
            for j in range(NumberofTrainingData):
                result = max(1 - InputWeight[i, 0], P[0, j])
                
                for k in range(1, NumberofAtributos):
                    temp = max(1 - InputWeight[i, k], P[k, j])
                    
                    if temp < result:
                        result = temp
                        
                H[i, j] = result
    
    return H

def pesos_xai_auxiliar(nivel, InputWeightTotal, InputWeightClass, feature_saturation, 
                       NumberofInputNeurons, entrada, count_nivel):

    # Filtragem das entradas benignas e malignas
    indice_linhas_benigno = entrada[:, 0] == 1
    entrada_benigno = entrada[indice_linhas_benigno, :]

    indice_linhas_maligno = entrada[:, 0] == 2
    entrada_maligno = entrada[indice_linhas_maligno, :]
    # Calcula os valores mínimos das entradas benignas e malignas
    has_benign, has_malign = False, False
    if len(entrada_benigno != 0): 
        min_entrada_benigno = np.min(entrada_benigno[:, 1:], axis=0); has_benign = True
    else:
        min_entrada_benigno = np.zeros(NumberofInputNeurons, dtype=np.float64)

    if len(entrada_maligno != 0): 
        min_entrada_maligno = np.min(entrada_maligno[:, 1:], axis=0); has_malign = True
    else:
        min_entrada_maligno = np.zeros(NumberofInputNeurons, dtype=np.float64)
    
    min_entrada = np.vstack((min_entrada_benigno, min_entrada_maligno))

    print('##############################################################################################')
    print('Nível:', nivel)
    print('Count nível:', count_nivel)
    print('InputWeightTotal:', InputWeightTotal)
    print('Feature saturation:', feature_saturation)

    # Classe 1 - Benigno
    #print('Maligno:')
    print('Entrada benigno:', entrada_benigno)
    if len(entrada_benigno) > len(entrada_maligno):
        classe = 1
        if has_benign:
            InputWeightTotal = np.vstack((InputWeightTotal,
                                np.zeros((1, NumberofInputNeurons), dtype=np.float64) ) )
            InputWeightClass = np.append(InputWeightClass,[0])

            InputWeightTotal, InputWeightClass, feature_saturation, count_nivel = \
                pesos_xai_classe_por_classe(nivel, InputWeightTotal, InputWeightClass, 
                                            feature_saturation, NumberofInputNeurons, 
                                            entrada_benigno, classe, count_nivel, min_entrada)
            
            if InputWeightClass[-1] == 0 and nivel > 1:
                #Remove all added zeros in InputWeightTotal and InputWeightClass
                InputWeightTotal = InputWeightTotal[:-1]
                InputWeightClass = InputWeightClass[:-1]

        print('&&&&&&&&&&&&&&&')
        #print('Benigno:')
        print('InputWeightTotal:', InputWeightTotal)
        print('Entrada maligno:', entrada_maligno)
        
        if has_malign:
            classe += 1
            InputWeightTotal = np.vstack((InputWeightTotal,
                                np.zeros((1, NumberofInputNeurons), dtype=np.float64) ) )
            InputWeightClass = np.append(InputWeightClass,[0])

            InputWeightTotal, InputWeightClass, feature_saturation, count_nivel = \
                pesos_xai_classe_por_classe(nivel, InputWeightTotal, InputWeightClass, 
                                            feature_saturation, NumberofInputNeurons, 
                                            entrada_maligno, classe, count_nivel, min_entrada)
            if InputWeightClass[-1] == 0 and nivel > 1:
                #Remove all added zeros in InputWeightTotal and InputWeightClass
                InputWeightTotal = InputWeightTotal[:-1]
                InputWeightClass = InputWeightClass[:-1]
    else:
        classe = 2
        if has_malign:
            InputWeightTotal = np.vstack((InputWeightTotal,
                                np.zeros((1, NumberofInputNeurons), dtype=np.float64) ) )
            InputWeightClass = np.append(InputWeightClass,[0])
            
            InputWeightTotal, InputWeightClass, feature_saturation, count_nivel = \
                pesos_xai_classe_por_classe(nivel, InputWeightTotal, InputWeightClass, 
                                            feature_saturation, NumberofInputNeurons, 
                                            entrada_maligno, classe, count_nivel, min_entrada)
            if InputWeightClass[-1] == 0 and nivel > 1:
                #Remove all added zeros in InputWeightTotal and InputWeightClass
                InputWeightTotal = InputWeightTotal[:-1]
                InputWeightClass = InputWeightClass[:-1]
        
        if has_benign:
            classe -=1
            InputWeightTotal = np.vstack((InputWeightTotal,
                                np.zeros((1, NumberofInputNeurons), dtype=np.float64) ) )
            InputWeightClass = np.append(InputWeightClass,[0])

            InputWeightTotal, InputWeightClass, feature_saturation, count_nivel = \
                pesos_xai_classe_por_classe(nivel, InputWeightTotal, InputWeightClass, 
                                            feature_saturation, NumberofInputNeurons, 
                                            entrada_benigno, classe, count_nivel, min_entrada)
            if InputWeightClass[-1] == 0 and nivel > 1:
                #Remove all added zeros in InputWeightTotal and InputWeightClass
                InputWeightTotal = InputWeightTotal[:-1]
                InputWeightClass = InputWeightClass[:-1]

    print('InputWeightTotal:', InputWeightTotal)
    print('Feature saturation:', feature_saturation)
    # Reordenando para que os primeiros pesos sejam da classe e o segundo peso da contra-classe

    return InputWeightTotal, InputWeightClass, feature_saturation, count_nivel

def pesos_xai_classe_por_classe(nivel, InputWeightTotal, InputWeightClass, feature_saturation, 
                                NumberofInputNeurons, entrada, classe, count_nivel, min_entrada):

    # Verifica se a saturação para a classe já ocorreu
    if np.sum(feature_saturation[classe-1, :]) == feature_saturation.shape[1]:
        return InputWeightTotal, InputWeightClass, feature_saturation, count_nivel

    # Calcula os máximos
    vectorInput, inputMax = calcula_maximos(feature_saturation, entrada, NumberofInputNeurons, classe, min_entrada)
    ii = NumberofInputNeurons
    count_vector = np.ones(NumberofInputNeurons, dtype=np.float64)

    while ii > 0:
        feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii = pesos_xai_estudo(
            vectorInput, inputMax, feature_saturation, InputWeightTotal, InputWeightClass, 
            entrada, nivel, classe, ii, count_nivel, count_vector, min_entrada
        )
        ii -= 1

    # Rotina de teste
    linha_com_valor_unico = False

    for i in range(InputWeightTotal.shape[0]):
        if np.all(InputWeightTotal[i, :] == InputWeightTotal[i, 0]):
            linha_com_valor_unico = True
            linha = i
            break
    
    if linha_com_valor_unico:
        print('erro')
        print('InputWeightTotal:', InputWeightTotal)
        print('Feature saturation:', feature_saturation)
        print('Nível:', nivel)
        print('Classe:', classe)
        #input("Pressione qualquer tecla para continuar...")

    # Verifica se a saturação completa não ocorreu
    if np.sum(feature_saturation[classe-1, :]) < feature_saturation.shape[1]:
        count_nivel += 1
        print('***********************************************************')
        print('Classe:', classe)
        print('Feature saturation:', feature_saturation)
        print('Soma da saturação para a classe:', np.sum(feature_saturation[classe-1, :]))
        print('Tamanho da segunda dimensão da saturação:', feature_saturation.shape[1])
        print('Count nível:', count_nivel)

    return InputWeightTotal, InputWeightClass, feature_saturation, count_nivel

def pesos_xai_estudo(vectorInput, inputMax, feature_saturation, InputWeightTotal, InputWeightClass, 
                     entrada, nivel, classe, ii, count_nivel, count_vector, min_entrada):
    
    if nivel == 1:
        feature_saturation, InputWeightTotal, InputWeightClass, ii = update_pesos(
            vectorInput, feature_saturation, InputWeightTotal, InputWeightClass, 
            entrada, ii, classe, count_nivel
        )
        return feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii

    modas = listModes(entrada, ii)
    
    while True:
        present = np.isin(inputMax, InputWeightTotal[:, ii-1])
        if present:
            print('========Repetido===============')
            print('Classe:', classe)
            print('InputMax:', inputMax)
            print('Modas:', modas)
            print('ii:', ii)
            
            if feature_saturation[classe-1, ii-1]:
                print('**********************************************************')
                InputWeightTotal[count_nivel-1, ii-1] = min_entrada[classe-1, ii-1]
                inputMax = calcula_maximo_neuronio_por_neuronio(
                    vectorInput, feature_saturation[classe-1, :]
                )
                print('InputWeightTotal:', InputWeightTotal)
                print('InputMax:', inputMax)
                print('Decrementar')
                print('ii:', ii)
                print('Nivel:', nivel)
                print('Count Nivel:', count_nivel)
                return feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii
            
            count_vector[ii-1] += 1
            print('Count Vector[ii]:', count_vector[ii-1])
            
            if count_vector[ii-1] > len(modas):
                print('entrouentrouentrouentrouentrouentrouentrouentrou')
                feature_saturation[classe-1, ii-1] = 1
                print('Feature Saturation:', feature_saturation)
                
                if np.sum(feature_saturation[classe-1, :]) >= feature_saturation.shape[1]:
                    print('saturou totallllllllllllllllllllllllllllllllllllll')
                    print('Size InputWeightTotal:', InputWeightTotal.shape)
                    print('InputWeightTotal:', InputWeightTotal)
                    print('Count Nivel:', count_nivel)
                    
                    if InputWeightTotal.shape[0] >= count_nivel:
                        InputWeightTotal = np.delete(InputWeightTotal, count_nivel-1, axis=0)
                        InputWeightClass = np.delete(InputWeightClass, count_nivel-1)
                    
                    count_nivel -= 1
                    print('InputWeightTotal:', InputWeightTotal)
                    ii = 0
                    return feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii
                else:
                    print('saturou parcial')
                    vectorInput, inputMax = calcula_maximos_saturado(
                        vectorInput, feature_saturation, 
                        entrada, InputWeightTotal.shape[1], classe, 
                        min_entrada, ii
                    )
                    ii = vectorInput.shape[0] + 1
                    print('VectorInput:', vectorInput)
                    print('InputMax:', inputMax)
                    return feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii
            
            print('****Substituicao************')
            print('VectorInput:', vectorInput)
            print('InputMax:', inputMax)
            vectorInput, inputMax = calcula_maximos_moda(
                vectorInput, feature_saturation, modas, count_vector, 
                entrada, InputWeightTotal.shape[1], classe, min_entrada, ii
            )
            ii = vectorInput.shape[0] + 1
            print('VectorInput:', vectorInput)
            print('InputMax:', inputMax)
            return feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii
        else:
            break

    print('%%%%%%%%%%%%%%%%')
    print('Classe:', classe)
    print('Count Nivel:', count_nivel)
    print('Feature Saturation:', feature_saturation)
    print('VectorInput:', vectorInput)
    feature_saturation, InputWeightTotal, InputWeightClass, ii = update_pesos(
        vectorInput, feature_saturation, InputWeightTotal, InputWeightClass, 
        entrada, ii, classe, count_nivel
    )

    return feature_saturation, InputWeightTotal, InputWeightClass, vectorInput, inputMax, count_vector, ii

def update_pesos(vectorInput, feature_saturation, InputWeightTotal, InputWeightClass, 
                 entrada, ii, classe, count_nivel):
    # Atualiza os pesos de entrada e a classe correspondente
    #TODO: Corrigir essa função
    InputWeightTotal[count_nivel-1][ii-1] = vectorInput[ii-1]
    InputWeightClass[count_nivel-1] = classe

    return feature_saturation, InputWeightTotal, InputWeightClass, ii

def calcula_maximos_saturado(vectorInput, feature_saturation, entrada, NumberofInputNeurons, 
                             classe, min_entrada, index):
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('VectorInput antes:', vectorInput)
    vectorInput[index-1] = np.min(entrada[:, index-1])
    print('VectorInput depois:', vectorInput)
    print('Entrada:', entrada)

    linhasSelecionadas = entrada[entrada[:, index-1] == vectorInput[index-1], :]
    print('Linhas Selecionadas:', linhasSelecionadas)
    print('Min Entrada:', min_entrada)

    vectorInput, inputMax, index = calcula_maximos_auxiliar(
        feature_saturation, linhasSelecionadas, NumberofInputNeurons, classe, min_entrada
    )
    
    linhasSelecionadas = entrada[entrada[:, index-1] == vectorInput[index-1], :]
    print('VectorInput final:', vectorInput)
    print('InputMax:', inputMax)

    #input("Pressione qualquer tecla para continuar...")

    return vectorInput, inputMax

def calcula_maximos_moda(vectorInput, feature_saturation, modas, count_vector, 
                         entrada, NumberofInputNeurons, classe, min_entrada, index):
    vectorInput[index-1] = modas[int(count_vector[index-1]-1)]
    linhasSelecionadas = entrada[entrada[:, index-1] == vectorInput[index-1], :]
    
    vectorInput, inputMax, index = calcula_maximos_auxiliar(
        feature_saturation, linhasSelecionadas, NumberofInputNeurons, classe, min_entrada
    )

    return vectorInput, inputMax

def calcula_maximos(feature_saturation, entrada, NumberofInputNeurons, classe, min_entrada):
    # Chama a função auxiliar para calcular os máximos iniciais
    vectorInput, inputMax, index = calcula_maximos_auxiliar(
        feature_saturation, entrada, NumberofInputNeurons, classe, min_entrada
    )
    # Filtra as linhas onde o valor na coluna index é igual ao inputMax
    linhasSelecionadas = entrada[entrada[:, index] == inputMax, :]

    # Chama a função auxiliar novamente com as linhas filtradas
    vectorInput, inputMax, index = calcula_maximos_auxiliar(
        feature_saturation, linhasSelecionadas, NumberofInputNeurons, classe, min_entrada
    )

    return vectorInput, inputMax

def calcula_maximos_auxiliar(feature_saturation, entrada, NumberofInputNeurons, classe, min_entrada):
    
    vectorInput = np.zeros(NumberofInputNeurons)
    entrada = entrada[:, 1:]
    for ii in range(NumberofInputNeurons-1, -1, -1):
        if feature_saturation[classe - 1, ii] == 0:
            vectorInput[ii] = stats.mode(entrada[:, ii])[0]  # Calcula a moda da coluna
        else:
            vectorInput[ii] = np.min(entrada[:, ii])  # Rotina de teste
    
    # Filtra o vetorInput, excluindo os valores onde feature_saturation[classe, :] é 1
    filtrado = vectorInput[feature_saturation[classe - 1, :] != 1]
    
    # Determina o valor máximo do vetor filtrado
    inputMax = np.max(filtrado)
    index = np.where(vectorInput == inputMax)[0][0] + 1
    return vectorInput, inputMax, index

def calcula_maximo_neuronio_por_neuronio(A, B):
    # Encontra os índices onde B é igual a zero
    indices = B == 0

    # Seleciona os elementos de A correspondentes aos índices onde B é zero
    valores_filtrados = A[indices]

    # Encontra o maior valor entre os elementos filtrados
    maior_valor = np.max(valores_filtrados)

    # Exibe o resultado
    print(maior_valor)

    return maior_valor

def listModes(entrada, ii):
    # Extrai a coluna correspondente de 'entrada'
    coluna = entrada[:, ii-1]
    
    # Conta as frequências dos valores únicos
    frequencias = Counter(coluna)
    
    # Ordena os valores únicos por suas frequências em ordem decrescente
    modas_ordenadas = sorted(frequencias.items(), key=lambda x: -x[1])
    
    # Extrai as modas (os valores únicos ordenados)
    modas = np.array([item[0] for item in modas_ordenadas])
    #if len(set(frequencias.values())) == 1 and len(frequencias) > 1:
    #    modas = sorted(modas, reverse=True)
    return modas

def rodaELMxaiEtapa(nivel, u, kernel, c_index, custo, g_index, gamma, strb,
                    numEntradas, classes, conjuntoTreinamento,
                    InputWeight, NumberofHiddenNeurons,
                    e_index, iteracao, fold, classificador, etapa):
    
    # Transposição das entradas e saídas de treinamento
    entradasTreinamento = conjuntoTreinamento[:, 1:].T

    saidasTreinamento = conjuntoTreinamento[:, 0].T

    # Conjunto de teste
    conjuntoTeste = conjuntoTreinamento
    entradasTeste = conjuntoTeste[:, 1:].T
    saidasTeste = conjuntoTeste[:, 0].T
    # Preparação dos dados de entrada
    REGRESSION, CLASSIFIER, T, P, TVT, TVP, C, NumberofTrainingData, NumberofTestingData, Elm_Type = \
        dados_entrada_xai(entradasTreinamento, saidasTreinamento, entradasTeste, saidasTeste, custo, kernel, strb)
    
    # Exibe o tamanho das matrizes antes da poda
    print('******Antes da Poda**************')
    print('Size T:', T.shape)
    print('Size P:', P.shape)
    
    # Treinamento ELM
    (T, P, 
     train_accuracy, test_accuracy,
       TrainingTime, TestingTime) = elm_autoral_xai(classificador, Elm_Type, kernel,
                           InputWeight, NumberofHiddenNeurons,
                           iteracao, fold, e_index, c_index, g_index,
                           REGRESSION, CLASSIFIER, T, P, TVT, TVP, C,
                           NumberofTrainingData, NumberofTestingData,
                           etapa)
    print('Train Accuracy:', train_accuracy)
    print('Test Accuracy:', test_accuracy)  
    # Atualiza o conjunto de treinamento com os dados podados
    conjuntoTreinamento = P
    print('******Poda**************')
    print('Size conjuntoTreinamento:', conjuntoTreinamento.shape)
    print('Size T:', T.shape)
    # Atualiza as classes no conjunto de treinamento
    indice_linhas = (T[0, :] == 1)  # Classe
    conjuntoTreinamento = np.vstack([conjuntoTreinamento, np.zeros((1, conjuntoTreinamento.shape[1]))])
    conjuntoTreinamento[-1, indice_linhas] = 1
    
    print('Size conjuntoTreinamento após classe:', conjuntoTreinamento.shape)
    
    indice_linhas = (T[1, :] == 1)  # Contra-classe
    conjuntoTreinamento[-1, indice_linhas] = 2
    conjuntoTreinamento = conjuntoTreinamento.T

    #Colocando a última coluna na primeira
    conjuntoTreinamento = np.roll(conjuntoTreinamento, 1, axis=1) 
    print('Size conjuntoTreinamento final:', conjuntoTreinamento.shape)
    print('Nível:', nivel)
    print('------------------------')

    # Verifica se o conjunto de treinamento está vazio
    break_flag = 0
    if conjuntoTreinamento.size == 0:
        break_flag = 1

    return conjuntoTreinamento, break_flag


def dados_entrada_xai(entradasTreinamento, saidasTreinamento, entradasTeste, saidasTeste, Regularization_coefficient, kernel, strb):
    
    Elm_Type = 1

    REGRESSION = 0
    CLASSIFIER = 1

    # Load training dataset
    if kernel in ['dilatacao_bitwise', 'erosao_bitwise']:
        if strb == 'int8':
            P = entradasTreinamento.astype(np.int8)
            TVP = entradasTeste.astype(np.int8)
        elif strb == 'int16':
            P = entradasTreinamento.astype(np.int16)
            TVP = entradasTeste.astype(np.int16)
        elif strb == 'int32':
            P = entradasTreinamento.astype(np.int32)
            TVP = entradasTeste.astype(np.int32)
    else:
        P = entradasTreinamento
        TVP = entradasTeste

    T = saidasTreinamento

    # Load testing dataset
    TVT = saidasTeste

    C = Regularization_coefficient
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

    return REGRESSION, CLASSIFIER, T, P, TVT, TVP, C, NumberofTrainingData, NumberofTestingData, Elm_Type

def elm_autoral_xai(classificador, Elm_Type, ActivationFunction, 
                    InputWeight, NumberofHiddenNeurons, 
                    iteracao, fold, e_index, c_index, g_index, 
                    REGRESSION, CLASSIFIER, T, P, TVT, TVP, C, 
                    NumberofTrainingData, NumberofTestingData, 
                    etapa):
    
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
        train_accuracy, T, P = avaliacaoRedeELM_XAI(NumberofTrainingData, Y, T, P, etapa, 'treino')
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



def avaliacaoRedeELM_XAI(numTeste, saidasRede, saidasDesejada, entrada, etapa, treino):
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
        if etapa == 1:
            mask = nodoVencedorRede == nodoVencedorDesejado
            saidasDesejada = saidasDesejada[:, ~mask]
            entrada = entrada[:, ~mask]

    return accuracy, saidasDesejada, entrada

# TODO: Implement this function
def confusao_funcao_elm(T, Y, TVT, TY, iteracao, ActivationFunction, e_index, c_index, g_index, classificador, fold):
    # Implementation should go here
    pass

def normalizar_pesos(matriz):
    vetor = matriz.flatten()
    maximo = np.max(vetor)
    minimo = np.min(vetor)

    ra = 0.8
    rb = 0.2

    R = (((ra - rb) * (matriz - minimo)) / (maximo - minimo)) + rb
    
    return R

def avaliarRedeTreino(Y, NumberofTestingData, T):
    # Get the index of the maximum output (winner neuron) for each pattern
    nodoVencedorRede = np.argmax(Y, axis=0)
    nodoVencedorDesejado = np.argmax(T, axis=0)

    # Count the number of misclassifications
    classificacoesErradas = np.sum(nodoVencedorRede != nodoVencedorDesejado)
    wrongIndexes = np.where(nodoVencedorRede != nodoVencedorDesejado)
    # Calculate accuracy
    accuracy = 1 - (classificacoesErradas / NumberofTestingData)
    accuracy = round(accuracy * 100, 2)
    
    return accuracy, wrongIndexes

def avaliarRede(TY, TVP):
    # Encontra o valor máximo e o índice do vencedor
    nodoVencedorRede = np.argmax(TY, axis=0)
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for padrao in range(TVP.shape[1]):
        if nodoVencedorRede[padrao] == 0:  # Em Python, o índice começa em 0
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

def grafico_auxiliar_Rildo(author, ActivationFunction, pasta,
                           NumberofTrainingData, NumberofTestingData,
                           InputWeight, InputWeightClass, P, T, TVP,
                           benignInput, malignInput):
    if ActivationFunction.lower() == 'dilatacao_classica': ActivationFunction = 'dilation'
    NumberofHiddenNeurons = InputWeight.shape[0]
    BiasMatrix =  np.zeros((NumberofHiddenNeurons, 1))
    #----------------------------------------------------------------------

    H = switchActivationFunction(ActivationFunction, InputWeight, BiasMatrix, P)
    
    OutputWeight = np.linalg.pinv(H.T) @ T.T
    Y = (H.T @ OutputWeight).T  
    del H
    
    acc, wrongIndexes = avaliarRedeTreino(Y, NumberofTrainingData, T)
    print(P[:,wrongIndexes].T)

    H_test = switchActivationFunction(ActivationFunction, InputWeight, BiasMatrix, TVP)
    
    TY = (H_test.T @ OutputWeight).T   
    del H_test
    
    xx1, yy1, xx2, yy2 = avaliarRede(TY, TVP)
    #----------------------------------------------------------------------
    plotar_Rildo(author, pasta, ActivationFunction, NumberofHiddenNeurons,
                 InputWeight, InputWeightClass,
                 benignInput, malignInput, xx1, yy1, xx2, yy2, acc)

def grafico_auxiliar(author, ActivationFunction, pasta, NumberofHiddenNeurons,
                     NumberofTrainingData, NumberofTestingData,
                     InputWeight, P, T, TVP,
                     BiasofHiddenNeurons,
                     entrada_benigno, entrada_maligno):

    # Calculate the hidden layer output matrix (H)
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

    # Plot the results
    plotar(author, pasta, ActivationFunction, NumberofHiddenNeurons,
           entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc)

def grafico_nohidden_auxiliar(author, pasta, ActivationFunction, NumberofHiddenNeurons,
                              NumberofTrainingData, NumberofTestingData,
                              InputWeight, P, T, TVP,
                              BiasofHiddenNeurons,
                              entrada_benigno, entrada_maligno):
    
    # Calcula a matriz tempH
    tempH = np.dot(InputWeight, P)
    BiasMatrix = np.tile(BiasofHiddenNeurons, (1, NumberofTrainingData))
    tempH += BiasMatrix

    # Escolhe a função de ativação
    if ActivationFunction.lower() in ['sig', 'sigmoid']:
        H = 1 / (1 + np.exp(-tempH))
        OutputWeight = np.dot(pinv(H.T), T.T)
        Y = np.dot(H.T, OutputWeight).T
    elif ActivationFunction.lower() in ['sin', 'sine']:
        H = np.sin(tempH)
        OutputWeight = np.dot(pinv(H.T), T.T)
        Y = np.dot(H.T, OutputWeight).T
    elif ActivationFunction.lower() == 'radbas':
        H = np.exp(-tempH**2)  # Assuming radbas is Gaussian RBF
        OutputWeight = np.dot(pinv(H.T), T.T)
        Y = np.dot(H.T, OutputWeight).T
    elif ActivationFunction.lower() == 'linear':
        Omega_train = np.dot(P.T, P)
        n = T.shape[1]
        OutputWeight = np.linalg.solve(Omega_train + np.eye(n)/1, T.T)
        Y = np.dot(Omega_train, OutputWeight).T
    elif ActivationFunction.lower() in ['fuzzy-dilation', 'fuzzy-erosion']:
        H = switchActivationFunction(ActivationFunction,InputWeight, BiasMatrix, P)
        OutputWeight = np.dot(pinv(H.T), T.T)
        Y = np.dot(H.T, OutputWeight).T
    
    # Avalia a rede de treinamento
    acc, wrongIndexes = avaliarRedeTreino(Y, NumberofTrainingData, T)
    
    # Calcula a matriz tempH_test
    tempH_test = np.dot(InputWeight, TVP)
    BiasMatrix = np.tile(BiasofHiddenNeurons, (1, NumberofTestingData))
    tempH_test += BiasMatrix
    
    # Escolhe a função de ativação para os dados de teste
    if ActivationFunction.lower() in ['sig', 'sigmoid']:
        H_test = 1 / (1 + np.exp(-tempH_test))
    elif ActivationFunction.lower() in ['sin', 'sine']:
        H_test = np.sin(tempH_test)
    elif ActivationFunction.lower() == 'radbas':
        H_test = np.exp(-tempH_test**2)  # Assuming radbas is Gaussian RBF
    elif ActivationFunction.lower() == 'linear':
        H_test = np.dot(P.T, TVP)
    elif ActivationFunction.lower() in ['fuzzy-dilation', 'fuzzy-erosion']:
        H_test = switchActivationFunction(ActivationFunction, InputWeight, BiasMatrix, TVP)
    
    # Calcula a saída para os dados de teste
    TY = np.dot(H_test.T, OutputWeight).T
    
    # Avalia a rede para os dados de teste
    xx1, yy1, xx2, yy2 = avaliarRede(TY, TVP)
    
    # Plota os resultados
    plotar(author, pasta, ActivationFunction, NumberofHiddenNeurons,
           entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc)

def grafico_xai(num_amostras, *args):
    NumberofHiddenNeurons = 100
    if 'sigmoid' in args: grafico_xai_inter('sigmoid', NumberofHiddenNeurons, num_amostras)
    if 'linear' in args: grafico_xai_inter('linear', NumberofHiddenNeurons, num_amostras)
    if 'radbas' in args: grafico_xai_inter('radbas', NumberofHiddenNeurons, num_amostras)
    if 'sine' in args: grafico_xai_inter('sine', NumberofHiddenNeurons,num_amostras)
    print(args)
    #grafico_xai_inter('authoral', NumberofHiddenNeurons,num_amostras)
    #grafico_xai_inter('antivirus', NumberofHiddenNeurons,num_amostras)
    


def grafico_xai_inter(kernel, NumberofHiddenNeurons,num_amostras):
    if kernel == 'antivirus':
        entrada_maligno = [
    [-4.2308, -0.7077, 2.0000],
    [-3.7179, -0.5795, 2.0000],
    [-3.4615, -0.5654, 2.0000],
    [-2.6923, -0.3731, 2.0000],
    [-2.4359, -0.2590, 2.0000],
    [-1.9231, -0.1808, 2.0000],
    [-0.6410, 0.1897, 2.0000],
    [-0.1282, 0.2679, 2.0000],
    [0.1282, 0.3821, 2.0000],
    [0.3846, 0.4462, 2.0000],
    [1.1538, 0.6385, 2.0000],
    [1.4103, 0.6526, 2.0000],
    [2.1795, 0.8949, 2.0000],
    [2.6923, 0.9731, 2.0000],
    [2.9487, 1.0872, 2.0000],
    [3.4615, 1.1654, 2.0000],
    [3.7179, 1.2295, 2.0000],
    [4.2308, 1.3577, 2.0000],
    [4.7436, 1.4859, 2.0000],
    [5.0000, 1.6000, 2.0000],
    [-0.6410, 0.1258, 2.0000],
    [-0.6410, -0.4492, 2.0000],
    [-0.1282, 0.1429, 2.0000],
    [-0.1282, -0.2688, 2.0000],
    [0.1282, 0.3046, 2.0000],
    [0.1282, -0.3313, 2.0000],
    [0.3846, 0.3671, 2.0000],
    [0.3846, -0.2262, 2.0000],
    [1.1538, 0.3885, 2.0000],
    [1.1538, -0.1068, 2.0000],
    [1.4103, 0.5226, 2.0000],
    [1.4103, -0.0653, 2.0000],
    [2.1795, 0.6565, 2.0000],
    [2.1795, 0.0821, 2.0000],
    [2.1795, -0.5191, 2.0000],
    [2.6923, 0.8361, 2.0000],
    [2.6923, 0.1414, 2.0000],
    [2.6923, -0.4009, 2.0000],
    [2.9487, 0.9121, 2.0000],
    [2.9487, 0.2071, 2.0000],
    [2.9487, -0.3080, 2.0000],
    [3.4615, 0.7439, 2.0000],
    [3.4615, 0.3336, 2.0000],
    [3.4615, -0.2236, 2.0000],
    [3.7179, 0.9973, 2.0000],
    [3.7179, 0.3186, 2.0000],
    [3.7179, -0.1810, 2.0000],
    [4.2308, 1.2098, 2.0000],
    [4.2308, 0.4533, 2.0000],
    [4.2308, -0.0996, 2.0000],
    [4.7436, 0.8352, 2.0000],
    [4.7436, 0.4906, 2.0000],
    [4.7436, -0.0291, 2.0000],
    [5.0000, 0.8321, 2.0000],
    [5.0000, 0.4731, 2.0000],
    [5.0000, 0.0344, 2.0000],
    [5.0000, -0.6090, 2.0000]
]
        entrada_benigno = [
    [-5.0000, -0.4500, 1.0000],
    [-4.4872, -0.3718, 1.0000],
    [-3.4615, -0.1154, 1.0000],
    [-2.6923, 0.1269, 1.0000],
    [-2.4359, 0.1410, 1.0000],
    [-1.9231, 0.3192, 1.0000],
    [-1.6667, 0.3833, 1.0000],
    [-1.4103, 0.4474, 1.0000],
    [-0.6410, 0.5897, 1.0000],
    [-0.1282, 0.7179, 1.0000],
    [0.1282, 0.7821, 1.0000],
    [0.3846, 0.8962, 1.0000],
    [0.6410, 0.9603, 1.0000],
    [1.1538, 1.0885, 1.0000],
    [1.4103, 1.1026, 1.0000],
    [2.1795, 1.2949, 1.0000],
    [2.6923, 1.4731, 1.0000],
    [2.9487, 1.5372, 1.0000],
    [4.4872, 1.9218, 1.0000],
    [4.7436, 1.9859, 1.0000],
    [-5.0000, 0.5224, 1.0000],
    [-5.0000, 1.6459, 1.0000],
    [-4.4872, 0.5410, 1.0000],
    [-4.4872, 1.7133, 1.0000],
    [-3.4615, 0.8727, 1.0000],
    [-3.4615, 1.9073, 1.0000],
    [-2.6923, 0.1600, 1.0000],
    [-2.6923, 1.1520, 1.0000],
    [-2.4359, 0.1794, 1.0000],
    [-2.4359, 1.4195, 1.0000],
    [-1.9231, 0.3765, 1.0000],
    [-1.9231, 1.4374, 1.0000],
    [-1.6667, 0.5440, 1.0000],
    [-1.6667, 1.7356, 1.0000],
    [-1.4103, 0.6539, 1.0000],
    [-1.4103, 1.7192, 1.0000],
    [-0.6410, 0.7125, 1.0000],
    [-0.6410, 1.6834, 1.0000],
    [-0.1282, 1.0353, 1.0000],
    [0.1282, 1.1112, 1.0000],
    [0.3846, 1.2486, 1.0000],
    [0.6410, 1.3524, 1.0000],
    [1.1538, 1.4001, 1.0000],
    [1.4103, 1.2429, 1.0000],
    [2.1795, 1.4656, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000]
]
        
        df1 = pd.DataFrame(entrada_benigno).astype('float64')
        df2 = pd.DataFrame(entrada_maligno).astype('float64')
        dSet = pd.concat([df1, df2], ignore_index=True)
        dSet = dSet.reset_index(drop=True)
        dSet = dSet[[2,0,1]].reset_index(drop=True)
        dSet.columns = range(dSet.shape[1])
        benignInput = dSet[dSet[0] == 1].drop(columns=0).to_numpy()
        malignInput = dSet[dSet[0] == 2].drop(columns=0).to_numpy()
        iteracao = 1
        NumberofInputNeurons = benignInput.shape[1]
        P = dSet.drop(columns=0).to_numpy().T
    elif kernel != "authoral":
        #[x1, y1, x2, y2, x11, y11, x22, y22] = distribuicao(kernel,num_amostras)
        [x1,y1,x2,y2,x11,y11,x22,y22] = data_from_matlab(kernel)
        iteracao = 1
        NumberofInputNeurons = 2
        # Stack arrays vertically (equivalent to MATLAB's vertcat)
        tempa = np.vstack((x1, x11))
        tempb = np.vstack((y1, y11))
        # Stack arrays horizontally (equivalent to MATLAB's horzcat)
        benignInput = np.hstack((tempa, tempb))

        tempa = np.vstack((x2, x22))
        tempb = np.vstack((y2, y22))
        malignInput = np.hstack((tempa, tempb))
        # Stack benigno and maligno entries vertically and transpose (equivalent to vertcat and transpose)
        P = np.vstack((benignInput, malignInput)).T
    elif kernel == 'authoral':
        benignInput, malignInput, benignTest, malignTest = create_classification_data(num_samples=num_amostras)
        iteracao = 1
        NumberofInputNeurons = benignInput.shape[1]
        P = np.vstack((benignInput, malignInput))   
        TP = np.vstack((benignTest, malignTest))
    
    #Pesos para as execuções padrão
    InputWeight, BiasofHiddenNeurons = pesos(iteracao, NumberofHiddenNeurons, NumberofInputNeurons)

    #--------------------------------------------------------------------RILDO
    entrada_benigno_Rildo = copy.deepcopy(benignInput)
    entrada_maligno_Rildo = copy.deepcopy(malignInput)

    #insert value of 1 in the first column of the benigno matrix
    entrada_benigno_Rildo = np.insert(entrada_benigno_Rildo, 0, 1, axis=1)
    #insert value of 2 in the first column of the maligno matrix
    entrada_maligno_Rildo = np.insert(entrada_maligno_Rildo, 0, 2, axis=1)

    conjuntoTreinamento = np.vstack((entrada_maligno_Rildo, entrada_benigno_Rildo))
    
    InputWeight_xai_Rildo, InputWeightClass = pesos_xai_rildo(
        iteracao, NumberofInputNeurons, conjuntoTreinamento
    )
    #---------------------------------------------------------------------------- RILDO



    # Initialize T with ones
    T = np.ones((2, P.shape[1]))

    # Set values in T based on the condition
    T[1, :benignInput.shape[0]] = -1
    T[0, benignInput.shape[0]:] = -1

    # Calculate minimum and maximum values for P
    if kernel != "authoral":
        minP1 = np.min(P[0, :])
        maxP1 = np.max(P[0, :])
        minP2 = np.min(P[1, :])
        maxP2 = np.max(P[1, :])
        # Create vectors using linspace
        vetora = np.linspace(minP1, maxP1, int(160))
        vetorb = np.linspace(minP2, maxP2, int(80))
        # Create the combinatorial matrix similar to combvec in MATLAB
        TVP = np.array(list(product(vetora, vetorb))).T
    else:
        if False: #don't
            minPs = np.min(P, axis=1)
            maxPs = np.max(P, axis=1)
            #vectors = np.linspace(minPs, maxPs, num_amostras)
            #FOR RILDO TVP
            corr_df = pd.DataFrame(columns=['r', 'p-value'])
            InputWeightsCalc = InputWeight_xai_Rildo
            #Add InputWeightClass to the InputWeightsCalc, in column 0
            InputWeightsCalc = np.insert(InputWeightsCalc, 0, InputWeightClass, axis=1)
            for col in range(InputWeightsCalc.shape[1]):
                if pd.api.types.is_numeric_dtype(InputWeightsCalc[col]):
                    r, p = stats.pearsonr(InputWeightsCalc[0], InputWeightsCalc[col])
                    corr_df.loc[col] = [round(r, 3), round(p, 3)]
            indices = sorted(corr_df['p-value'].items(), key=lambda x: x[1])[1:]
            chosenIndices = [index for index, _ in indices] #always only two indices
            #vetora = np.linspace(minPs[int(chosenIndices[0])], maxPs[int(chosenIndices[0])], int(num_amostras*3.2))
            #vetorb = np.linspace(minPs[int(chosenIndices[1])], maxPs[int(chosenIndices[1])], int(num_amostras*1.6))
            vectors = np.linspace(minPs, maxPs, num_amostras)
        else:
            TVP = TP.T

    #make TVP including all indices in chosenIndices product
    #TVP = np.array(list(product(*vectors[chosenIndices]))).T
    
    NumberofTestingData = TVP.shape[1]
    NumberofTrainingData = P.shape[1]
    
    #LINEAR
    # grafico_auxiliar('PINHEIRO','linear', kernel, NumberofHiddenNeurons,
    #                  NumberofTrainingData, NumberofTestingData,
    #                  InputWeight, P, T, TVP,
    #                  BiasofHiddenNeurons,
    #                  benignInput, malignInput)

    # ELM Clássica
    # grafico_auxiliar('PINHEIRO','dilation', kernel, NumberofHiddenNeurons,
    #                      NumberofTrainingData, NumberofTestingData,
    #                      InputWeight, P, T, TVP,
    #                      BiasofHiddenNeurons,
    #                      benignInput, malignInput)
    
    # grafico_auxiliar('PINHEIRO','erosao_classica', kernel, NumberofHiddenNeurons,
    #                     NumberofTrainingData,NumberofTestingData,
    #                     InputWeight, P, T, TVP,
    #                     BiasofHiddenNeurons,
    #                     benignInput, malignInput)
    #ELM sem camada escondida
    # grafico_nohidden_auxiliar('HUANG',kernel, kernel, NumberofHiddenNeurons,
    #                         NumberofTrainingData,NumberofTestingData, 
    #                         InputWeight, P, T, TVP,
    #                         BiasofHiddenNeurons,
    #                         benignInput, malignInput)

    # grafico_nohidden_auxiliar('HUANG', kernel, 'fuzzy-erosion', NumberofHiddenNeurons,
    #                             NumberofTrainingData,NumberofTestingData, 
    #                             InputWeight, P, T, TVP,
    #                             BiasofHiddenNeurons,
    #                             benignInput, malignInput) 
    #ELM XAI
    #ELM XAI Rildo
    grafico_auxiliar_Rildo('XAIRildo','dilatacao_classica', kernel,
                         NumberofTrainingData, NumberofTestingData,
                         InputWeight_xai_Rildo, InputWeightClass, P, T, TVP,
                         benignInput, malignInput)
    

num_amostras =  40
x = grafico_xai(num_amostras)