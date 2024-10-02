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
def plotar(author, pasta, ActivationFunction, NumberofHiddenNeurons, 
           entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc, 
           InputWeight=None, InputWeightClass=None):

    # Create scatter plot for xx1, yy1 (blue 'x') and xx2, yy2 (red 'o')
    fig = go.Figure()
    
    # Plot testing data points
    fig.add_trace(go.Scatter(x=xx1, y=yy1, mode='markers',
                             marker=dict(size=points_size, color='blue', symbol='x'),
                             name='Class 1', opacity=0.6))
    
    fig.add_trace(go.Scatter(x=xx2, y=yy2, mode='markers',
                             marker=dict(size=points_size, color='red', symbol='circle'),
                             name='Class 2',opacity=0.6))
    
    # Handling InputWeight and InputWeightClass similar to plotar_Rildo
    if InputWeight is not None and InputWeightClass is not None:
        # Find the indices and filter benign and malignant entries
        indice_linhas_1 = np.where(InputWeightClass == 1)[0]
        x3 = InputWeight[indice_linhas_1, 0]
        y3 = InputWeight[indice_linhas_1, 1]

        indice_linhas_2 = np.where(InputWeightClass == 2)[0]
        x4 = InputWeight[indice_linhas_2, 0]
        y4 = InputWeight[indice_linhas_2, 1]

        # Remove duplicated lines of benign and malignant entries
        entrada_benigno_filtered = entrada_benigno[~np.isin(entrada_benigno, np.vstack((x3, y3)).T).all(axis=1)]
        entrada_maligno_filtered = entrada_maligno[~np.isin(entrada_maligno, np.vstack((x4, y4)).T).all(axis=1)]

        x1 = entrada_benigno_filtered[:, 0]
        y1 = entrada_benigno_filtered[:, 1]

        x2 = entrada_maligno_filtered[:, 0]
        y2 = entrada_maligno_filtered[:, 1]
    else:
        # If no InputWeight or InputWeightClass, just plot the data
        x1 = entrada_benigno[:, 0]
        y1 = entrada_benigno[:, 1]

        x2 = entrada_maligno[:, 0]
        y2 = entrada_maligno[:, 1]

    # Plot training data points for benign and malignant
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers',
                             marker=dict(size=samples_size, color='blue', symbol='x', line=dict(width=1.25)),
                             name='Benigno'))

    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers',
                             marker=dict(size=samples_size, color='red', symbol='circle', line=dict(width=1.25)),
                             name='Maligno'))

    # Add text over points if InputWeight is provided
    if InputWeight is not None and InputWeightClass is not None:
        for i in range(len(x3)):
            fig.add_trace(go.Scatter(
                x=[x3[i]], y=[y3[i]],
                mode='text',
                text=[str(indice_linhas_1[i])],
                textposition='middle center',
                textfont=dict(size=text_size, color='blue'),
                showlegend=False
            ))

        for i in range(len(x4)):
            fig.add_trace(go.Scatter(
                x=[x4[i]], y=[y4[i]],
                mode='text',
                text=[str(indice_linhas_2[i])],
                textposition='middle center',
                textfont=dict(size=text_size, color='red'),
                showlegend=False
            ))

    # Adjust axis appearance
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0), 
        autosize=True,
        title=f'{author}|{ActivationFunction}|{NumberofHiddenNeurons} Hidden Neurons|Accuracy: {acc:.2f}%',
        title_x=0.5,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        showlegend=False
    )

    # Display the figure
    #fig.show()
    if not os.path.exists("ELM_XAI"):
        os.mkdir("images")
    if not os.path.exists(f"ELM_XAI/{pasta}"):
        os.mkdir(f"ELM_XAI/{pasta}")
    #fig.write_image(f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png')
    pio.write_image(fig, f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png',scale=2, width=864, height=720)

def plotar_Rildo(author, pasta, ActivationFunction, NumberofHiddenNeurons,
                 InputWeight, InputWeightClass, 
                 entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc):
    # points_size = 5
    # samples_size = 10
    # text_size = 30
    # Criação da figura
    fig = go.Figure()

    # Scatter plot para xx1, yy1 e xx2, yy2
    fig.add_trace(go.Scatter(
        x=xx1, y=yy1,
        mode='markers',
        marker=dict(symbol='x', size=points_size, color='blue'),
        name='Class 1',
        opacity=0.6
    ))

    fig.add_trace(go.Scatter(
        x=xx2, y=yy2,
        mode='markers',
        marker=dict(symbol='circle', size=points_size, color='red'),
        name='Class 2',
        opacity=0.6
    ))

    # Encontrar os índices e filtrar entradas benignas e malignas
    indice_linhas_1 = np.where(InputWeightClass == 1)[0]
    x3 = InputWeight[indice_linhas_1, 0]
    y3 = InputWeight[indice_linhas_1, 1]

    indice_linhas_2 = np.where(InputWeightClass == 2)[0]
    x4 = InputWeight[indice_linhas_2, 0]
    y4 = InputWeight[indice_linhas_2, 1]

    # Remover as linhas duplicadas de entrada_benigno e entrada_maligno
    entrada_benigno_filtered = entrada_benigno[~np.isin(entrada_benigno, np.vstack((x3, y3)).T).all(axis=1)]
    entrada_maligno_filtered = entrada_maligno[~np.isin(entrada_maligno, np.vstack((x4, y4)).T).all(axis=1)]

    x1 = entrada_benigno_filtered[:, 0]
    y1 = entrada_benigno_filtered[:, 1]

    x2 = entrada_maligno_filtered[:, 0]
    y2 = entrada_maligno_filtered[:, 1]

    # Scatter plot para entradas benignas e malignas
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        mode='markers',
        marker=dict(symbol='x', size=samples_size, color='blue', line=dict(width=1.25)),
        name='Benigno'
    ))

    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='markers',
        marker=dict(symbol='circle', size=samples_size, color='red', line=dict(width=1.25)),
        name='Maligno'
    ))

    # Adicionar textos sobre os pontos
    for i in range(len(x3)):
        fig.add_trace(go.Scatter(
            x=[x3[i]], y=[y3[i]],
            mode='text',
            text=[str(indice_linhas_1[i])],
            textposition='middle center',
            textfont=dict(size=text_size, color='blue'),
            showlegend=False
        ))

    for i in range(len(x4)):
        fig.add_trace(go.Scatter(
            x=[x4[i]], y=[y4[i]],
            mode='text',
            text=[str(indice_linhas_2[i])],
            textposition='middle center',
            textfont=dict(size=text_size, color='red'),
            showlegend=False
        ))

    # Ajuste da aparência dos eixos
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Atualizar o layout com título e outros ajustes
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0), 
        autosize=True,
        title=f'{author}|{ActivationFunction}|{NumberofHiddenNeurons} Hidden Neurons|Accuracy: {acc:.2f}%',  # Adicionar título com precisão,
        title_x=0.5,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        showlegend=False
    )

    # Exibir a figura
    #fig.show()
    if not os.path.exists("ELM_XAI"):
        os.mkdir("images")
    if not os.path.exists(f"ELM_XAI/{pasta}"):
        os.mkdir(f"ELM_XAI/{pasta}")
    #fig.write_image(f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png')
    pio.write_image(fig, f'ELM_XAI/{pasta}/{author}_{ActivationFunction}_{acc}_{NumberofHiddenNeurons}.png',scale=2, width=864, height=720)

def pesos(iteracao, NumberofHiddenNeurons, NumberofInputNeurons):
    # Set the seed for reproducibility
    np.random.seed(iteracao)
    
    # Generate random weights between -1 and 1
    InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1
    
    # Generate random biases
    np.random.seed(iteracao)
    BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons, 1)
    
    # Normalize weights and biases
    InputWeight = normalizar_pesos(InputWeight)
    BiasofHiddenNeurons = normalizar_pesos(BiasofHiddenNeurons)
    
    return InputWeight, BiasofHiddenNeurons

def pesos_xai_rildo(iteracao, NumberofInputNeurons, conjuntoTreinamento):
    classes = 2
    u = 1
    kernel = 'dilatacao_classica'
    if kernel == 'dilatacao_classica': kernel = 'dilation'
    c_index = 1 
    c = 1 
    g_index = 1 
    g = 1 
    e_index = 1
    fold = 1
    classificador = 'ELM_XAI'
    strb = np.int32
    conjuntoTreinamentoELM = conjuntoTreinamento
    features_atomico = np.ones(NumberofInputNeurons, dtype=np.float64)
    feature_saturation = np.zeros((2, NumberofInputNeurons), dtype=np.float64)
    InputWeight = np.zeros((classes, NumberofInputNeurons), dtype=np.float64)
    InputWeightClass = np.zeros((classes), dtype=np.float64)
    nivel = 0
    count_nivel = 1

    while True:
        # Incrementa o nível
        nivel += 1

        InputWeight, InputWeightClass, feature_saturation, count_nivel = pesos_xai_auxiliar(
            nivel, InputWeight, InputWeightClass, feature_saturation, 
            NumberofInputNeurons, conjuntoTreinamento, count_nivel
        )
        if nivel == 1:
            #Remove the zeroes from the InputWeight matrix
            InputWeight = InputWeight[~np.all(InputWeight == 0, axis=1)]
            InputWeightClass = InputWeightClass[InputWeightClass != 0]
        
        conjuntoTreinamento, break_flag = rodaELMxaiEtapa(
            nivel, u, kernel, c_index, c, g_index, g, strb,
            NumberofInputNeurons, classes, conjuntoTreinamentoELM,
            InputWeight, len(InputWeight),
            e_index, iteracao, fold, classificador, 1
        )


        if break_flag == 1:
            break
        
        if len(set(conjuntoTreinamento[:, 0])) == 1 and ({1} in [set(f) for f in feature_saturation]): # 
            break
        # Verifica se todas as features estão saturadas
        if np.all(feature_saturation == 1):
            print('cima saturou')
            break

    
    InputWeight = np.vstack((InputWeight[np.where(InputWeightClass == 1)], 
                                       InputWeight[np.where(InputWeightClass == 2)]))
    InputWeightClass = np.concatenate((InputWeightClass[np.where(InputWeightClass == 1)], 
                                       InputWeightClass[np.where(InputWeightClass == 2)]))
    
    return InputWeight, InputWeightClass

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

def sine_distr(y1, x1, y2, x2):
    # Adicionando valores a y1 e x1
    y1 = np.append(y1, [0.5, 0.45, 0.4, 0.2, 0.3, -0.5, -0.45, -0.4, -0.3, -0.2])
    x1 = np.append(x1, [-2, -1.5, -1.0, -0.75, -0.5, -1.5, -1.25, -1.5, -1.75, -1.5])

    # Adicionando valores a y2 e x2
    y2 = np.append(y2, [-1, -1, -1, -1, -1.1, -1.2, -1.2, -1.2, -1, -1, -1, -1, -1.1, -1.2, -1.2, -1.2, -1.05, -1.05, -1.05, -1.05, -1.05, -1.05, -1.05, -1.05])
    x2 = np.append(x2, [-6, -5, -4, -3, -6, -5, -4, -3, 0, 1, 2, 2.5, 0, 1, 2, 2.5, 0, 1, 2, 2.5, 0, 1, 2, 2.5])

    return y1, x1, y2, x2

def sigmoid_distr(y1, x1, y2, x2):
    # Adicionando valores a y1 e x1
    y1 = np.append(y1, [0.7, 0.7, 0.7, 0.95, 0.95, 0.95, 0.95, 0.95, 0.7, 0.7, 
                        0.90, 0.90, 1.0, 0.90, 0.8, 0.85, 1.0, 0.7, 1.0])
    x1 = np.append(x1, [-9, -7, -3, -8, -6, -2, -1, 0, -9.5, -4.5, 
                        -4, -1.5, 0.1, -4.5, -3.0, -2.0, -4.85, -5.55, -7.5])
    
    # Adicionando valores a y2 e x2
    y2 = np.append(y2, [-0.05, -0.1])
    x2 = np.append(x2, [-2.5, 10])

    return y1, x1, y2, x2

def radbas_distr(y1, x1, y2, x2):
    # Adicionando valores a y1 e x1
    y1 = np.append(y1, [
        0.6, 0.6, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.1, 1.1, 
        1.1, 1.0, 1.0, 0.6, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 
        1.2, 1.2, 1.2, 1.2, 1.1, 1.2, 1.2, 1.2, 1.2, 1.2, 
        1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8, 0.5, 0.5, 
        0.5, 1.1, 1.1, 0.9
    ])
    
    x1 = np.append(x1, [
        -5, 5, -4, 4, 4.75, 5, -4.25, -2, -5, -4, 
        -3, 3, 5, -4.75, -4.75, -4.25, -3.75, -0.5, 4.85, -2.85, 
        -1.85, 1.85, -1.75, -4.25, -4, -3, -2, 1.5, 4.0, 4.25, 
        1.25, -1.25, -0.55, 2.0, 3.0, 4.0, 4.5, 2.0, 3.0, 4.0, 
        4.5, -4.5, 1.5, 4.75
    ])

    return y1, x1, y2, x2

def dime(entrada):
    # Verifica se a entrada é um array 1D e converte para 2D se necessário
    if len(entrada.shape) == 1:
        entrada = entrada[:, np.newaxis]  # Converte para vetor coluna
    
    # Verifica se a entrada é uma única linha, se sim, transposta
    elif entrada.shape[0] == 1:
        entrada = entrada.T
    return entrada

def preencher_espacos_min(y, x):
    y_min = np.min(y)
    y_max = np.max(y)
    dd = (y_max - y_min) / 3
    
    # Cria uma matriz com valores interpolados entre y_min e y_max
    y_grid = np.arange(y_min, y_max + dd, dd)
    
    # Repete cada valor de x, y_grid.shape[0] vezes (um para cada valor interpolado)
    x_expanded = np.repeat(x, y_grid.shape[0])
    y_expanded = np.tile(y_grid, len(x))
    
    # Gera o ruído
    rr = (0.1 * y_expanded) + (0.2 * y_expanded) * np.random.rand(len(y_expanded))
    
    # Adiciona o ruído aos valores de y
    ya = y_expanded + rr
    
    # Filtra os valores que estão fora dos limites de y
    valid_indices = (ya >= y_min) & (ya <= y_max) & (ya >= np.repeat(y, y_grid.shape[0]))
    
    # Seleciona apenas os valores válidos
    ya = ya[valid_indices]
    xa = x_expanded[valid_indices]
    
    return ya, xa

def preencher_espacos_max(y, x):
    y_min = np.min(y)
    y_max = np.max(y)
    dd = -(y_max - y_min) / 3
    
    # Cria uma matriz com valores interpolados entre y_max e y_min, decrementando por dd
    y_grid = np.arange(y_max, y_min + dd, dd)
    
    # Repete cada valor de x, y_grid.shape[0] vezes (um para cada valor interpolado)
    x_expanded = np.repeat(x, y_grid.shape[0])
    y_expanded = np.tile(y_grid, len(x))
    
    # Gera o ruído
    rr = (0.1 * y_expanded) + (0.2 * y_expanded) * np.random.rand(len(y_expanded))
    
    # Subtrai o ruído dos valores de y
    ya = y_expanded - rr
    
    # Filtra os valores que estão fora dos limites de y
    valid_indices = (ya <= y_max) & (ya >= y_min) & (ya <= np.repeat(y, y_grid.shape[0]))
    
    # Seleciona apenas os valores válidos
    ya = ya[valid_indices]
    xa = x_expanded[valid_indices]
    
    return ya, xa

def adicionar_ruido_peq_neg(yy, xx, ruido):
    for ii in range(len(yy)-1, -1, -1):
        rr = np.random.randint(-ruido, 1) / 20.0
        yy[ii] += rr
    return yy, xx

def adicionar_ruido_peq_pos(yy, xx, ruido):
    for ii in range(len(yy)-1, -1, -1):
        rr = np.random.randint(0, ruido+1) / 20.0
        yy[ii] += rr
    return yy, xx

def adicionar_ruido_neg(yy, xx, ruido):
    for ii in range(len(yy)-1, -1, -1):
        rr = np.random.randint(-ruido, 1) / 20.0
        yy[ii] += rr
        if yy[ii] < -2:
            yy[ii] += 1.5
        elif yy[ii] > 2:
            yy[ii] -= 1.5
    return yy, xx

def adicionar_ruido_pos(yy, xx, ruido):
    for ii in range(len(yy)-1, -1, -1):
        rr = np.random.randint(0, ruido+1) / 20.0
        yy[ii] += rr
        if yy[ii] < -2:
            yy[ii] += 1.5
        elif yy[ii] > 2:
            yy[ii] -= 1.5
    return yy, xx

def eliminar_partes(yy, xx, perc):
    n = len(yy)
    # Calculando o número de elementos a serem removidos (perc% da amostra)
    num_remover = round(perc * n)
    
    # Gerando índices aleatórios para remover
    indices_remover = np.random.choice(n, num_remover, replace=False)
    
    # Removendo os elementos da amostra
    yy = np.delete(yy, indices_remover)
    xx = np.delete(xx, indices_remover)
    
    return yy, xx


def distribuicao(dist, num_amostras):
    # Definindo o número de amostras
    np.random.seed(2)

    if dist == 'sine':
        x1 = np.arange(-2 * np.pi, 2 * np.pi, 0.4)
        y1 = np.sin(x1)
        x2 = x1.copy()
        y2 = y1.copy()
        y1 += 0.2
        y2 -= 0.2
        y1, x1 = adicionar_ruido_peq_neg(y1, x1, 2)
        
        y1a = y1 + 0.05
        y1 = np.concatenate((y1a, y1))
        x1 = np.concatenate((x1, x1))

        y2, x2 = adicionar_ruido_peq_pos(y2, x2, 2)
        y2a = y2 - 0.05
        y2b = y2 - 0.1
        y2c = y2 - 0.15
        y2 = np.concatenate((y2a, y2))
        y2 = np.concatenate((y2b, y2))
        y2 = np.concatenate((y2c, y2))
        x2 = np.concatenate((x2, x2))
        x2 = np.concatenate((x2, x2))

    elif dist == 'sigmoid':
        x1 = np.linspace(-10, 10, num_amostras)
        y1 = 1 / (1 + np.exp(-x1)) + 0.02 * np.random.randn(num_amostras)
        y1 = np.clip(y1, 0, 1)
        x2 = x1.copy()
        y2 = y1 - 0.1

    elif dist == 'radbas':
        x1 = np.linspace(-5, 5, num_amostras)
        y1 = np.exp(-np.square(x1) / 2)
        y1 += 0.15
        x2 = x1.copy()
        y2 = y1 - 0.15
        y1, x1 = adicionar_ruido_peq_pos(y1, x1, 1)
        y2, x2 = adicionar_ruido_peq_neg(y2, x2, 1)

    else:
        coeficiente_angular = 0.25
        intercepto = 0.55
        x1 = np.linspace(-5, 5, num_amostras)
        y1 = coeficiente_angular * x1 + intercepto + 0.25
        x2 = x1.copy()
        y2 = coeficiente_angular * x2 + intercepto - 0.25
        y1, x1 = adicionar_ruido_peq_neg(y1, x1, 1)
        y2, x2 = adicionar_ruido_peq_pos(y2, x2, 1)
        y1, x1 = eliminar_partes(y1, x1, 0.5)
        y2, x2 = eliminar_partes(y2, x2, 0.5)

    y11, x11 = preencher_espacos_min(y1, x1)
    y22, x22 = preencher_espacos_max(y2, x2)

    mmin = min(len(y11), len(y22))
    mmax = max(len(y11), len(y22))

    for ii in range(mmin, mmax):
        if len(y11) == mmin:
            y11 = np.append(y11, y11[mmin - 1])
            x11 = np.append(x11, x11[mmin - 1])
        else:
            y22 = np.append(y22, y22[mmin - 1])
            x22 = np.append(x22, x22[mmin - 1])

    x1 = dime(x1)
    y1 = dime(y1)
    x11 = dime(x11)
    y11 = dime(y11)
    x2 = dime(x2)
    y2 = dime(y2)
    x22 = dime(x22)
    y22 = dime(y22)

    return x1, y1, x2, y2, x11, y11, x22, y22

def dime(arr):
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def preencher_espacos_min(y, x):
    y_min = np.min(y)
    y_max = np.max(y)
    dd = (y_max - y_min) / 3
    xa, ya = [], []

    for i in range(len(y)):
        j = y[i]
        while j <= y_max:
            mmin = 0.1 * j
            mmax = 0.2 * j
            rr = mmin + (mmax + mmax) * np.random.rand()
            ya.append(j + rr)
            xa.append(x[i])
            if ya[-1] > y_max:
                ya.pop()
                xa.pop()
            j += dd

    return np.array(ya), np.array(xa)

def preencher_espacos_max(y, x):
    y_min = np.min(y)
    y_max = np.max(y)
    dd = -(y_max - y_min) / 3
    xa, ya = [], []

    for i in range(len(y)):
        j = y[i]
        while j >= y_min:
            mmin = 0.1 * j
            mmax = 0.2 * j
            rr = mmin + (mmax + mmax) * np.random.rand()
            ya.append(j - rr)
            xa.append(x[i])
            if ya[-1] < y_min:
                ya.pop()
                xa.pop()
            j += dd

    return np.array(ya), np.array(xa)

def adicionar_ruido_peq_neg(y, x, ruido):
    rr = np.random.randint(-ruido, 0, size=len(y)) / 20
    return y + rr, x

def adicionar_ruido_peq_pos(y, x, ruido):
    rr = np.random.randint(0, ruido, size=len(y)) / 20
    return y + rr, x

def eliminar_partes(y, x, perc):
    n = len(y)
    num_remover = round(perc * n)
    indices_remover = random.sample(range(n), num_remover)
    y = np.delete(y, indices_remover)
    x = np.delete(x, indices_remover)
    return y, x

def create_classification_data(num_samples=100, num_features=10, random_state=42):
    # Criando o dataset de classificação com 10 features e 2 classes (benign e malign)
    num_samples = num_samples*2
    X, y = make_classification(n_samples=num_samples, n_features=num_features, 
                               n_informative=5, n_redundant=2, n_clusters_per_class=1,
                               n_classes=2, flip_y=0.01, random_state=random_state)
    
    # Transformando as classes 0 e 1 em 'benign' e 'malign'
    y = np.where(y == 0, 'benign', 'malign')
    benignInput = X[y == 'benign']
    benignTest = X[y == 'benign']
    benignInput, benignTest = benignInput[:num_samples//2], benignTest[num_samples//2:]
    malignInput = X[y == 'malign']
    malignTest = X[y == 'malign']
    malignInput, malignTest = malignInput[:num_samples//2], malignTest[num_samples//2:]

    return benignInput, malignInput, benignTest, malignTest

def data_from_matlab(kernel):
    if kernel == 'sigmoid':
        string = """x1 =

  -10.0000
   -9.4872
   -8.9744
   -8.4615
   -7.9487
   -7.4359
   -6.9231
   -6.4103
   -5.8974
   -5.3846
   -4.8718
   -4.3590
   -3.8462
   -3.3333
   -2.8205
   -2.3077
   -1.7949
   -1.2821
   -0.7692
   -0.2564
    0.2564
    0.7692
    1.2821
    1.7949
    2.3077
    2.8205
    3.3333
    3.8462
    4.3590
    4.8718
    5.3846
    5.8974
    6.4103
    6.9231
    7.4359
    7.9487
    8.4615
    8.9744
    9.4872
   10.0000
   -9.0000
   -7.0000
   -3.0000
   -8.0000
   -6.0000
   -2.0000
   -1.0000
         0
   -9.5000
   -4.5000
   -4.0000
   -1.5000
    0.1000
   -4.5000
   -3.0000
   -2.0000
   -4.8500
   -5.5500
   -7.5000


y1 =

         0
         0
    0.0057
         0
         0
         0
         0
    0.0055
         0
         0
    0.0183
    0.0148
         0
    0.0354
    0.0315
    0.1033
    0.1654
    0.2169
    0.3299
    0.3853
    0.5640
    0.6630
    0.7805
    0.8420
    0.8867
    0.9502
    0.9541
    0.9465
    0.9686
    0.9868
    0.9937
    0.9796
    1.0000
    0.9976
    0.9998
    0.9918
    1.0000
    1.0000
    0.9879
    1.0000
    0.7000
    0.7000
    0.7000
    0.9500
    0.9500
    0.9500
    0.9500
    0.9500
    0.7000
    0.7000
    0.9000
    0.9000
    1.0000
    0.9000
    0.8000
    0.8500
    1.0000
    0.7000
    1.0000


x2 =

  -10.0000
   -9.4872
   -8.9744
   -8.4615
   -7.9487
   -7.4359
   -6.9231
   -6.4103
   -5.8974
   -5.3846
   -4.8718
   -4.3590
   -3.8462
   -3.3333
   -2.8205
   -2.3077
   -1.7949
   -1.2821
   -0.7692
   -0.2564
    0.2564
    0.7692
    1.2821
    1.7949
    2.3077
    2.8205
    3.3333
    3.8462
    4.3590
    4.8718
    5.3846
    5.8974
    6.4103
    6.9231
    7.4359
    7.9487
    8.4615
    8.9744
    9.4872
   10.0000
   -2.5000
   10.0000


y2 =

   -0.1000
   -0.1000
   -0.0943
   -0.1000
   -0.1000
   -0.1000
   -0.1000
   -0.0945
   -0.1000
   -0.1000
   -0.0817
   -0.0852
   -0.1000
   -0.0646
   -0.0685
    0.0033
    0.0654
    0.1169
    0.2299
    0.2853
    0.4640
    0.5630
    0.6805
    0.7420
    0.7867
    0.8502
    0.8541
    0.8465
    0.8686
    0.8868
    0.8937
    0.8796
    0.9000
    0.8976
    0.8998
    0.8918
    0.9000
    0.9000
    0.8879
    0.9000
   -0.0500
   -0.1000


x11 =

  -10.0000
  -10.0000
  -10.0000
   -9.4872
   -9.4872
   -9.4872
   -8.9744
   -8.9744
   -8.9744
   -8.4615
   -8.4615
   -8.4615
   -7.9487
   -7.9487
   -7.9487
   -7.4359
   -7.4359
   -7.4359
   -6.9231
   -6.9231
   -6.9231
   -6.4103
   -6.4103
   -6.4103
   -5.8974
   -5.8974
   -5.8974
   -5.3846
   -5.3846
   -5.3846
   -4.8718
   -4.8718
   -4.8718
   -4.3590
   -4.3590
   -4.3590
   -3.8462
   -3.8462
   -3.8462
   -3.3333
   -3.3333
   -3.3333
   -2.8205
   -2.8205
   -2.8205
   -2.3077
   -2.3077
   -1.7949
   -1.7949
   -1.2821
   -1.2821
   -0.7692
   -0.7692
   -0.2564
    0.2564
    0.7692
    1.2821
   -9.0000
   -7.0000
   -3.0000
   -9.5000
   -4.5000
   -5.5500
   -5.5500
   -5.5500
   -5.5500
   -5.5500
   -5.5500


y11 =

         0
    0.4333
    0.9705
         0
    0.4237
    0.8498
    0.0075
    0.5022
    0.8859
         0
    0.4155
    0.9602
         0
    0.3996
    0.7512
         0
    0.4734
    0.8938
         0
    0.4057
    0.8731
    0.0062
    0.5060
    0.8581
         0
    0.4098
    0.8026
         0
    0.4649
    0.8345
    0.0260
    0.4247
    0.9131
    0.0165
    0.4752
    0.8553
         0
    0.4220
    0.8269
    0.0527
    0.4222
    0.8602
    0.0351
    0.5090
    0.9515
    0.1224
    0.5530
    0.2257
    0.5826
    0.3061
    0.6347
    0.4616
    0.9414
    0.4579
    0.7540
    0.7993
    0.9896
    0.7967
    0.8495
    0.8302
    0.7837
    0.8350
    0.7787
    0.7787
    0.7787
    0.7787
    0.7787
    0.7787


x22 =

   -2.3077
   -1.7949
   -1.2821
   -0.7692
   -0.2564
   -0.2564
    0.2564
    0.2564
    0.7692
    0.7692
    1.2821
    1.2821
    1.2821
    1.7949
    1.7949
    1.7949
    2.3077
    2.3077
    2.3077
    2.8205
    2.8205
    2.8205
    3.3333
    3.3333
    3.3333
    3.8462
    3.8462
    3.8462
    4.3590
    4.3590
    4.3590
    4.8718
    4.8718
    4.8718
    5.3846
    5.3846
    5.3846
    5.8974
    5.8974
    5.8974
    6.4103
    6.4103
    6.4103
    6.4103
    6.9231
    6.9231
    6.9231
    7.4359
    7.4359
    7.4359
    7.9487
    7.9487
    7.9487
    8.4615
    8.4615
    8.4615
    8.4615
    8.9744
    8.9744
    8.9744
    8.9744
    9.4872
    9.4872
    9.4872
   10.0000
   10.0000
   10.0000
   10.0000


y22 =

    0.0021
    0.0339
    0.0842
    0.1878
    0.2083
   -0.0347
    0.3234
    0.0901
    0.4972
    0.1916
    0.4899
    0.2141
    0.0081
    0.4372
    0.2856
    0.0390
    0.6036
    0.3207
    0.0721
    0.4734
    0.3795
    0.1029
    0.6168
    0.3196
    0.1681
    0.7533
    0.2688
    0.1545
    0.5518
    0.4211
    0.1656
    0.6604
    0.2932
    0.1479
    0.4766
    0.3469
    0.1588
    0.6136
    0.4438
    0.1912
    0.6596
    0.3823
    0.1282
   -0.0629
    0.5054
    0.2988
    0.1209
    0.5161
    0.4800
    0.1775
    0.7282
    0.3088
    0.1313
    0.6844
    0.3810
    0.1349
   -0.0567
    0.7359
    0.3444
    0.1350
   -0.0883
    0.7610
    0.4026
    0.1795
    0.4930
    0.3469
    0.1435
   -0.0592

        """
    elif kernel == 'linear':
        string = """x1 =

   -5.0000
   -4.4872
   -3.4615
   -3.2051
   -2.9487
   -2.6923
   -2.4359
   -1.6667
   -0.8974
   -0.1282
    0.1282
    0.6410
    0.8974
    1.4103
    2.1795
    3.2051
    3.7179
    3.9744
    4.4872
    5.0000


y1 =

   -0.4500
   -0.3218
   -0.1154
   -0.0013
    0.0128
    0.0769
    0.1410
    0.3333
    0.5256
    0.7679
    0.7821
    0.9103
    1.0244
    1.1026
    1.3449
    1.6013
    1.6795
    1.7436
    1.9218
    2.0000


x2 =

   -5.0000
   -2.9487
   -2.4359
   -2.1795
   -1.9231
   -1.1538
   -0.6410
   -0.3846
   -0.1282
    0.3846
    0.6410
    1.1538
    1.4103
    3.2051
    3.4615
    3.7179
    3.9744
    4.2308
    4.7436
    5.0000


y2 =

   -0.9500
   -0.4372
   -0.3090
   -0.1949
   -0.1808
    0.0115
    0.1897
    0.2538
    0.3179
    0.4462
    0.4603
    0.5885
    0.6526
    1.1513
    1.1654
    1.2295
    1.3436
    1.3577
    1.5359
    1.6000


x11 =

   -5.0000
   -5.0000
   -4.4872
   -4.4872
   -3.4615
   -3.4615
   -3.2051
   -3.2051
   -2.9487
   -2.9487
   -2.6923
   -2.6923
   -2.4359
   -2.4359
   -1.6667
   -1.6667
   -0.8974
   -0.8974
   -0.1282
   -0.1282
    0.1282
    0.6410
    0.8974
    1.4103
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795
    2.1795


y11 =

    0.4668
    1.5950
    0.7331
    1.6777
    0.8907
    1.9397
    1.0684
    1.8230
    0.0149
    1.0618
    0.1085
    1.2607
    0.2091
    1.1806
    0.4664
    1.6596
    0.7566
    1.7152
    1.0646
    1.7484
    0.8681
    1.0385
    1.2429
    1.3841
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861
    1.7861


x22 =

   -1.1538
   -1.1538
   -0.6410
   -0.6410
   -0.3846
   -0.3846
   -0.1282
   -0.1282
    0.3846
    0.3846
    0.6410
    0.6410
    1.1538
    1.1538
    1.4103
    1.4103
    3.2051
    3.2051
    3.2051
    3.4615
    3.4615
    3.4615
    3.7179
    3.7179
    3.7179
    3.9744
    3.9744
    3.9744
    4.2308
    4.2308
    4.2308
    4.7436
    4.7436
    4.7436
    5.0000
    5.0000
    5.0000
    5.0000


y22 =

    0.0065
   -0.4440
    0.0993
   -0.3787
    0.2151
   -0.4539
    0.2596
   -0.2942
    0.2601
   -0.3071
    0.3095
   -0.2253
    0.3335
   -0.2139
    0.3966
   -0.1142
    1.0163
    0.2582
   -0.3984
    0.9459
    0.1728
   -0.3273
    0.7559
    0.2247
   -0.3612
    1.0864
    0.3206
   -0.2617
    0.7132
    0.3265
   -0.2551
    0.8570
    0.5128
   -0.1360
    0.9397
    0.5333
   -0.0796
   -0.5919

"""
    elif kernel == 'radbas':
        string = """x1 =

   -5.0000
   -4.7436
   -4.4872
   -4.2308
   -3.9744
   -3.7179
   -3.4615
   -3.2051
   -2.9487
   -2.6923
   -2.4359
   -2.1795
   -1.9231
   -1.6667
   -1.4103
   -1.1538
   -0.8974
   -0.6410
   -0.3846
   -0.1282
    0.1282
    0.3846
    0.6410
    0.8974
    1.1538
    1.4103
    1.6667
    1.9231
    2.1795
    2.4359
    2.6923
    2.9487
    3.2051
    3.4615
    3.7179
    3.9744
    4.2308
    4.4872
    4.7436
    5.0000
   -5.0000
    5.0000
   -4.0000
    4.0000
    4.7500
    5.0000
   -4.2500
   -2.0000
   -5.0000
   -4.0000
   -3.0000
    3.0000
    5.0000
   -4.7500
   -4.7500
   -4.2500
   -3.7500
   -0.5000
    4.8500
   -2.8500
   -1.8500
    1.8500
   -1.7500
   -4.2500
   -4.0000
   -3.0000
   -2.0000
    1.5000
    4.0000
    4.2500
    1.2500
   -1.2500
   -0.5500
    2.0000
    3.0000
    4.0000
    4.5000
   -4.5000
    1.5000
    4.7500


y1 =

    0.2000
    0.1500
    0.2000
    0.2001
    0.1504
    0.2010
    0.1525
    0.2059
    0.1629
    0.1767
    0.2015
    0.2430
    0.3074
    0.3994
    0.5699
    0.6639
    0.8185
    0.9643
    1.0787
    1.1918
    1.1418
    1.1287
    0.9643
    0.8685
    0.7139
    0.5199
    0.4494
    0.3074
    0.2930
    0.2515
    0.1767
    0.1629
    0.2059
    0.1525
    0.1510
    0.1504
    0.1501
    0.2000
    0.1500
    0.1500
    0.6000
    0.6000
    1.1000
    1.1000
    1.1000
    1.1000
    1.0000
    1.0000
    1.1000
    1.1000
    1.1000
    1.0000
    1.0000
    0.6000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.1000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    1.2000
    0.8000
    0.5000
    0.5000
    0.5000
    1.1000
    1.1000
    0.9000


x2 =

   -5.0000
   -4.7436
   -4.4872
   -4.2308
   -3.9744
   -3.7179
   -3.4615
   -3.2051
   -2.9487
   -2.6923
   -2.4359
   -2.1795
   -1.9231
   -1.6667
   -1.4103
   -1.1538
   -0.8974
   -0.6410
   -0.3846
   -0.1282
    0.1282
    0.3846
    0.6410
    0.8974
    1.1538
    1.4103
    1.6667
    1.9231
    2.1795
    2.4359
    2.6923
    2.9487
    3.2051
    3.4615
    3.7179
    3.9744
    4.2308
    4.4872
    4.7436
    5.0000


y2 =

   -0.0500
    0.0000
   -0.0500
   -0.0499
    0.0004
    0.0010
   -0.0475
   -0.0441
   -0.0371
    0.0267
    0.0015
    0.0930
    0.1074
    0.1994
    0.3699
    0.4639
    0.6185
    0.8143
    0.9287
    0.9918
    0.9918
    0.9287
    0.7643
    0.6185
    0.4639
    0.3199
    0.2494
    0.1074
    0.0430
    0.0515
    0.0267
    0.0129
    0.0059
   -0.0475
   -0.0490
    0.0004
   -0.0499
    0.0000
    0.0000
    0.0000


x11 =

   -5.0000
   -5.0000
   -4.7436
   -4.7436
   -4.7436
   -4.4872
   -4.4872
   -4.2308
   -4.2308
   -4.2308
   -3.9744
   -3.9744
   -3.9744
   -3.7179
   -3.7179
   -3.4615
   -3.4615
   -3.2051
   -3.2051
   -2.9487
   -2.9487
   -2.9487
   -2.6923
   -2.6923
   -2.6923
   -2.4359
   -2.4359
   -2.4359
   -2.1795
   -2.1795
   -1.9231
   -1.9231
   -1.6667
   -1.6667
   -1.4103
   -1.4103
   -1.1538
   -0.8974
   -0.6410
    0.6410
    0.8974
    1.1538
    1.4103
    1.4103
    1.6667
    1.6667
    1.9231
    1.9231
    2.1795
    2.1795
    2.1795
    2.4359
    2.4359
    2.4359
    2.6923
    2.6923
    2.6923
    2.9487
    2.9487
    2.9487
    3.2051
    3.2051
    3.2051
    3.4615
    3.4615
    3.4615
    3.7179
    3.7179
    3.9744
    3.9744
    3.9744
    4.2308
    4.2308
    4.2308
    4.4872
    4.4872
    4.4872
    4.7436
    4.7436
    4.7436
    5.0000
    5.0000
    5.0000
   -5.0000
    5.0000
   -2.0000
   -4.7500
    2.0000
    3.0000
    3.0000
    4.0000
    4.0000
    4.5000
    4.5000


y11 =

    0.2666
    0.6106
    0.1883
    0.6494
    1.0761
    0.2481
    0.7263
    0.2292
    0.6741
    1.0052
    0.2098
    0.6820
    1.0084
    0.2546
    0.7480
    0.1782
    0.7300
    0.2375
    0.8047
    0.2313
    0.6757
    1.0255
    0.2592
    0.7041
    1.0857
    0.2429
    0.8082
    1.1430
    0.3198
    0.7966
    0.4148
    0.7696
    0.5174
    0.9885
    0.6830
    1.0806
    0.7597
    0.9037
    1.1746
    1.1709
    1.1191
    0.9422
    0.7775
    0.9739
    0.5360
    1.0850
    0.4451
    0.7803
    0.3988
    0.9128
    1.1047
    0.2997
    0.8311
    1.0799
    0.1965
    0.6546
    1.1711
    0.1826
    0.5777
    0.9642
    0.2590
    0.7601
    1.0682
    0.2212
    0.6397
    1.1490
    0.1836
    0.6744
    0.1924
    0.5918
    1.0801
    0.1918
    0.6517
    1.1139
    0.2234
    0.6412
    1.1521
    0.2075
    0.7055
    1.1992
    0.1952
    0.7414
    1.0478
    0.8396
    0.7595
    1.1409
    0.7969
    1.0621
    0.7166
    1.0050
    0.7107
    0.9497
    0.6370
    1.0101


x22 =

   -4.7436
   -3.9744
   -3.7179
   -2.6923
   -2.4359
   -2.1795
   -1.9231
   -1.6667
   -1.4103
   -1.4103
   -1.1538
   -1.1538
   -0.8974
   -0.8974
   -0.6410
   -0.6410
   -0.6410
   -0.3846
   -0.3846
   -0.3846
   -0.1282
   -0.1282
   -0.1282
   -0.1282
    0.1282
    0.1282
    0.1282
    0.1282
    0.3846
    0.3846
    0.3846
    0.6410
    0.6410
    0.6410
    0.8974
    0.8974
    1.1538
    1.1538
    1.4103
    1.4103
    1.6667
    1.9231
    2.1795
    2.4359
    2.6923
    2.9487
    3.2051
    3.9744
    4.4872
    4.7436
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000
    5.0000


y22 =

    0.0000
    0.0002
    0.0006
    0.0221
    0.0009
    0.0661
    0.0855
    0.1242
    0.1879
    0.0182
    0.2709
    0.0694
    0.5270
    0.1531
    0.5825
    0.3968
    0.0821
    0.7257
    0.4824
    0.2033
    0.8137
    0.4693
    0.1912
   -0.0432
    0.6087
    0.5539
    0.2467
   -0.0307
    0.5070
    0.2946
    0.1496
    0.5418
    0.3607
    0.0472
    0.5118
    0.1883
    0.3045
    0.0661
    0.2465
   -0.0138
    0.1252
    0.0612
    0.0268
    0.0312
    0.0147
    0.0067
    0.0033
    0.0002
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000
    0.0000


"""
    elif kernel == 'sine':
        string = """x1 =

   -6.2832
   -5.8832
   -5.4832
   -5.0832
   -4.6832
   -4.2832
   -3.8832
   -3.4832
   -3.0832
   -2.6832
   -2.2832
   -1.8832
   -1.4832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    0.9168
    1.3168
    1.7168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.5168
    4.9168
    5.3168
    5.7168
    6.1168
   -6.2832
   -5.8832
   -5.4832
   -5.0832
   -4.6832
   -4.2832
   -3.8832
   -3.4832
   -3.0832
   -2.6832
   -2.2832
   -1.8832
   -1.4832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    0.9168
    1.3168
    1.7168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.5168
    4.9168
    5.3168
    5.7168
    6.1168
   -2.0000
   -1.5000
   -1.0000
   -0.7500
   -0.5000
   -1.5000
   -1.2500
   -1.5000
   -1.7500
   -1.5000


y1 =

    0.1500
    0.5894
    0.9174
    1.0820
    1.1496
    1.0593
    0.8755
    0.4850
    0.0916
   -0.2425
   -0.6068
   -0.7516
   -0.8462
   -0.6335
   -0.4313
   -0.0294
    0.3665
    0.6441
    0.9937
    1.1179
    1.1894
    1.0546
    0.7349
    0.3729
    0.0257
   -0.3940
   -0.6778
   -0.7809
   -0.7792
   -0.6228
   -0.3866
    0.0344
    0.1000
    0.5394
    0.8674
    1.0320
    1.0996
    1.0093
    0.8255
    0.4350
    0.0416
   -0.2925
   -0.6568
   -0.8016
   -0.8962
   -0.6835
   -0.4813
   -0.0794
    0.3165
    0.5941
    0.9437
    1.0679
    1.1394
    1.0046
    0.6849
    0.3229
   -0.0243
   -0.4440
   -0.7278
   -0.8309
   -0.8292
   -0.6728
   -0.4366
   -0.0156
    0.5000
    0.4500
    0.4000
    0.2000
    0.3000
   -0.5000
   -0.4500
   -0.4000
   -0.3000
   -0.2000


x2 =

   -6.2832
   -5.8832
   -5.4832
   -5.0832
   -4.6832
   -4.2832
   -3.8832
   -3.4832
   -3.0832
   -2.6832
   -2.2832
   -1.8832
   -1.4832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    0.9168
    1.3168
    1.7168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.5168
    4.9168
    5.3168
    5.7168
    6.1168
   -6.2832
   -5.8832
   -5.4832
   -5.0832
   -4.6832
   -4.2832
   -3.8832
   -3.4832
   -3.0832
   -2.6832
   -2.2832
   -1.8832
   -1.4832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    0.9168
    1.3168
    1.7168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.5168
    4.9168
    5.3168
    5.7168
    6.1168
   -6.2832
   -5.8832
   -5.4832
   -5.0832
   -4.6832
   -4.2832
   -3.8832
   -3.4832
   -3.0832
   -2.6832
   -2.2832
   -1.8832
   -1.4832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    0.9168
    1.3168
    1.7168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.5168
    4.9168
    5.3168
    5.7168
    6.1168
   -6.2832
   -5.8832
   -5.4832
   -5.0832
   -4.6832
   -4.2832
   -3.8832
   -3.4832
   -3.0832
   -2.6832
   -2.2832
   -1.8832
   -1.4832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    0.9168
    1.3168
    1.7168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.5168
    4.9168
    5.3168
    5.7168
    6.1168
   -6.0000
   -5.0000
   -4.0000
   -3.0000
   -6.0000
   -5.0000
   -4.0000
   -3.0000
         0
    1.0000
    2.0000
    2.5000
         0
    1.0000
    2.0000
    2.5000
         0
    1.0000
    2.0000
    2.5000
         0
    1.0000
    2.0000
    2.5000


y2 =

   -0.3500
    0.1394
    0.4174
    0.6820
    0.7496
    0.6593
    0.3255
   -0.0150
   -0.4084
   -0.7425
   -1.0068
   -1.2516
   -1.3462
   -1.1835
   -0.8813
   -0.5794
   -0.1335
    0.1941
    0.4937
    0.6679
    0.6894
    0.6046
    0.2849
   -0.0271
   -0.4243
   -0.8940
   -1.1278
   -1.2309
   -1.2792
   -1.1228
   -0.8366
   -0.4656
   -0.3000
    0.1894
    0.4674
    0.7320
    0.7996
    0.7093
    0.3755
    0.0350
   -0.3584
   -0.6925
   -0.9568
   -1.2016
   -1.2962
   -1.1335
   -0.8313
   -0.5294
   -0.0835
    0.2441
    0.5437
    0.7179
    0.7394
    0.6546
    0.3349
    0.0229
   -0.3743
   -0.8440
   -1.0778
   -1.1809
   -1.2292
   -1.0728
   -0.7866
   -0.4156
   -0.2500
    0.2394
    0.5174
    0.7820
    0.8496
    0.7593
    0.4255
    0.0850
   -0.3084
   -0.6425
   -0.9068
   -1.1516
   -1.2462
   -1.0835
   -0.7813
   -0.4794
   -0.0335
    0.2941
    0.5937
    0.7679
    0.7894
    0.7046
    0.3849
    0.0729
   -0.3243
   -0.7940
   -1.0278
   -1.1309
   -1.1792
   -1.0228
   -0.7366
   -0.3656
   -0.2000
    0.2894
    0.5674
    0.8320
    0.8996
    0.8093
    0.4755
    0.1350
   -0.2584
   -0.5925
   -0.8568
   -1.1016
   -1.1962
   -1.0335
   -0.7313
   -0.4294
    0.0165
    0.3441
    0.6437
    0.8179
    0.8394
    0.7546
    0.4349
    0.1229
   -0.2743
   -0.7440
   -0.9778
   -1.0809
   -1.1292
   -0.9728
   -0.6866
   -0.3156
   -1.0000
   -1.0000
   -1.0000
   -1.0000
   -1.1000
   -1.2000
   -1.2000
   -1.2000
   -1.0000
   -1.0000
   -1.0000
   -1.0000
   -1.1000
   -1.2000
   -1.2000
   -1.2000
   -1.0500
   -1.0500
   -1.0500
   -1.0500
   -1.0500
   -1.0500
   -1.0500
   -1.0500


x11 =

   -6.2832
   -6.2832
   -5.8832
   -5.4832
   -3.8832
   -3.4832
   -3.0832
   -3.0832
   -2.6832
   -2.2832
   -2.2832
   -1.8832
   -1.8832
   -1.4832
   -1.4832
   -1.0832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    2.5168
    2.9168
    3.3168
    3.3168
    3.7168
    4.1168
    4.1168
    4.5168
    4.5168
    4.9168
    4.9168
    5.3168
    5.3168
    5.7168
    6.1168
    6.1168
   -6.2832
   -6.2832
   -5.8832
   -5.4832
   -3.8832
   -3.4832
   -3.0832
   -3.0832
   -2.6832
   -2.2832
   -2.2832
   -1.8832
   -1.8832
   -1.4832
   -1.4832
   -1.0832
   -1.0832
   -0.6832
   -0.2832
    0.1168
    0.5168
    2.1168
    2.5168
    2.9168
    3.3168
    3.7168
    4.1168
    4.1168
    4.5168
    4.5168
    4.9168
    4.9168
    5.3168
    5.3168
    5.7168
    6.1168
   -2.0000
   -1.5000
   -1.0000
   -0.7500
   -0.5000
   -1.5000
   -1.5000
   -1.2500
   -1.5000
   -1.7500
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000
   -1.5000


y11 =

    0.1826
    1.1069
    0.7324
    1.0259
    1.0763
    0.5839
    0.1313
    1.0973
    0.5003
    0.1178
    0.8699
   -0.0708
    0.8296
   -0.1873
    0.7185
    0.0707
    0.9274
    0.3682
    0.8433
    0.4976
    0.7524
    1.0285
    0.5293
    0.0305
    1.0578
    0.3730
    0.0220
    0.9377
   -0.1227
    0.8224
   -0.1123
    0.7919
    0.0998
    0.9197
    0.3531
    0.0380
    0.9862
    0.1218
    0.9343
    0.6139
    1.0525
    1.0894
    0.6255
    0.0466
    0.8789
    0.4689
    0.0536
    0.9985
   -0.1184
    0.7017
   -0.2281
    0.5496
    0.0131
    0.7962
    0.2691
    0.8932
    0.4030
    0.7225
    1.1883
    0.8699
    0.4126
    0.7493
    0.3215
   -0.0461
    0.8620
   -0.1674
    0.7231
   -0.1934
    0.7103
    0.0286
    0.9947
    0.2871
    0.9275
    0.6067
    0.5312
    0.5881
    0.2933
    0.3901
    0.2151
    1.1282
    0.3557
    0.4340
    0.4555
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165
    0.7165


x22 =

   -6.2832
   -5.8832
   -5.8832
   -5.4832
   -5.4832
   -5.4832
   -5.0832
   -5.0832
   -5.0832
   -4.6832
   -4.6832
   -4.6832
   -4.2832
   -4.2832
   -4.2832
   -3.8832
   -3.8832
   -3.8832
   -3.4832
   -3.0832
   -0.2832
    0.1168
    0.5168
    0.5168
    0.5168
    0.9168
    0.9168
    0.9168
    1.3168
    1.3168
    1.3168
    1.7168
    1.7168
    1.7168
    2.1168
    2.1168
    2.1168
    2.5168
    2.5168
    2.5168
    2.9168
    3.3168
    6.1168
   -6.2832
   -5.8832
   -5.8832
   -5.8832
   -5.4832
   -5.4832
   -5.4832
   -5.0832
   -5.0832
   -5.0832
   -4.6832
   -4.6832
   -4.6832
   -4.2832
   -4.2832
   -4.2832
   -3.8832
   -3.8832
   -3.8832
   -3.4832
   -3.4832
   -3.0832
   -0.2832
    0.1168
    0.5168
    0.5168
    0.5168
    0.9168
    0.9168
    0.9168
    1.3168
    1.3168
    1.3168
    1.7168
    1.7168
    1.7168
    2.1168
    2.1168
    2.1168
    2.5168
    2.5168
    2.5168
    2.9168
    2.9168
    3.3168
    6.1168
   -6.2832
   -5.8832
   -5.8832
   -5.8832
   -5.4832
   -5.4832
   -5.4832
   -5.0832
   -5.0832
   -5.0832
   -4.6832
   -4.6832
   -4.6832
   -4.2832
   -4.2832
   -4.2832
   -3.8832
   -3.8832
   -3.8832
   -3.4832
   -3.4832
   -3.0832
   -0.2832
    0.1168
    0.5168
    0.5168
    0.5168
    0.9168
    0.9168
    0.9168
    1.3168
    1.3168
    1.3168
    1.7168
    1.7168
    1.7168
    2.1168
    2.1168
    2.1168
    2.5168
    2.5168
    2.5168
    2.9168
    2.9168
    3.3168
    6.1168
   -6.2832
   -5.8832
   -5.8832
   -5.8832
   -5.4832
   -5.4832
   -5.4832
   -5.0832
   -5.0832
   -5.0832
   -4.6832
   -4.6832
   -4.6832
   -4.6832
   -4.2832
   -4.2832
   -4.2832
   -3.8832
   -3.8832
   -3.8832
   -3.4832
   -3.4832
   -3.0832
   -2.6832
   -0.2832
    0.1168
    0.1168
    0.5168
    0.5168
    0.5168
    0.9168
    0.9168
    0.9168
    1.3168
    1.3168
    1.3168
    1.7168
    1.7168
    1.7168
    2.1168
    2.1168
    2.1168
    2.5168
    2.5168
    2.5168
    2.9168
    2.9168
    3.3168
    6.1168


y22 =

   -0.8354
    0.0937
   -0.3521
    0.2365
   -0.2708
   -0.6562
    0.3947
   -0.0587
   -0.6987
    0.5442
    0.0008
   -0.4095
    0.4036
   -0.0549
   -0.4961
    0.2499
   -0.3421
   -0.7610
   -0.4011
   -0.8622
   -0.6744
   -0.5150
    0.1155
   -0.4724
   -0.7355
    0.3531
   -0.2166
   -0.6879
    0.5219
   -0.0669
   -0.7200
    0.5656
   -0.0431
   -0.5196
    0.5230
   -0.0884
   -0.7670
    0.2364
   -0.2850
   -0.6618
   -0.4955
   -1.0144
   -0.6095
   -0.6526
    0.1149
   -0.3075
   -0.6792
    0.2613
   -0.1810
   -0.8965
    0.4763
   -0.0125
   -0.6382
    0.4340
    0.0291
   -0.4809
    0.5069
   -0.0277
   -0.6269
    0.2774
   -0.2706
   -1.0028
    0.0190
   -0.6339
   -0.9538
   -0.6884
   -0.4270
    0.1545
   -0.3525
   -0.9383
    0.4159
   -0.1064
   -0.5937
    0.4928
   -0.0271
   -0.6242
    0.3865
   -0.0062
   -0.5736
    0.5867
   -0.0751
   -0.6726
    0.2786
   -0.3152
   -0.9518
    0.0200
   -0.6247
   -0.9793
   -0.9173
   -0.6234
    0.1538
   -0.3948
   -0.6891
    0.3857
   -0.1516
   -0.6768
    0.6022
    0.0177
   -0.4010
    0.4788
    0.0523
   -0.5822
    0.5339
    0.0056
   -0.6244
    0.3412
   -0.2255
   -0.6737
    0.0652
   -0.5159
   -0.9374
   -0.9208
   -0.6842
    0.2284
   -0.2794
   -0.6914
    0.4567
   -0.1258
   -0.6948
    0.3949
    0.0167
   -0.6090
    0.4912
    0.0344
   -0.5615
    0.4078
   -0.0343
   -0.5240
    0.2582
   -0.2887
   -0.8904
    0.0572
   -0.3714
   -0.9263
   -0.6524
   -0.6515
    0.1671
   -0.4069
   -1.0125
    0.3408
   -0.1352
   -0.7186
    0.5585
    0.0627
   -0.3566
    0.6704
    0.0854
   -0.3940
   -0.7205
    0.5717
    0.0314
   -0.4604
    0.4226
   -0.1651
   -0.5675
    0.1162
   -0.3158
   -0.6079
   -0.7021
   -0.8027
    0.0100
   -0.5510
    0.2162
   -0.2728
   -0.9418
    0.4385
   -0.0744
   -0.6528
    0.5193
    0.0607
   -0.5100
    0.6097
    0.0463
   -0.4083
    0.4903
    0.0050
   -0.5391
    0.3268
   -0.1643
   -0.7636
    0.0686
   -0.3995
   -0.9193
   -0.7725
"""
    #divide based on x1 =, x2 =, x11 =, x22 =, y1 =, y2 =, y11 =, y22 =
    string = re.split(r'x1 =|x2 =|x11 =|x22 =|y1 =|y2 =|y11 =|y22 =', string)[1:]
    y = []
    for st in string:
        x = st.split('\n')
        k = [float(i.strip()) for i in x if i.strip()]
        y.append(np.ndarray((1,len(k))))
        y[-1][0] = k
    x1,y1,x2,y2,x11,y11,x22,y22 = y
    return x1.T,y1.T,x2.T,y2.T,x11.T,y11.T,x22.T,y22.T

def distribuicao_alt(dist):
    if dist == 'sine':
        pass
        #x1, y1, x2, y2, x11, y11, x22, y22
    elif dist == 'sigmoid':
        #x1, y1, x2, y2, x11, y11, x22, y22
        pass
    elif dist == 'radbas':
        #x1, y1, x2, y2, x11, y11, x22, y22
        pass
    else:
        x1 =np.vstack(np.array([
                    -5.0000,
                    -4.4872,
                    -3.4615,
                    -3.2051,
                    -2.9487,
                    -2.6923,
                    -2.4359,
                    -1.6667,
                    -0.8974,
                    -0.1282,
                     0.1282,
                     0.6410,
                     0.8974,
                     1.4103,
                     2.1795,
                     3.2051,
                     3.7179,
                     3.9744,
                     4.4872,
                     5.0000,]))

        y1 =np.vstack(np.array([

                -0.4500,
                -0.3218,
                -0.1154,
                -0.0013,
                 0.0128,
                 0.0769,
                 0.1410,
                 0.3333,
                 0.5256,
                 0.7679,
                 0.7821,
                 0.9103,
                 1.0244,
                 1.1026,
                 1.3449,
                 1.6013,
                 1.6795,
                 1.7436,
                 1.9218,
                 2.0000,]))

        x2 =np.vstack(np.array([
                        -5.0000,
                        -2.9487,
                        -2.4359,
                        -2.1795,
                        -1.9231,
                        -1.1538,
                        -0.6410,
                        -0.3846,
                        -0.1282,
                         0.3846,
                         0.6410,
                         1.1538,
                         1.4103,
                         3.2051,
                         3.4615,
                         3.7179,
                         3.9744,
                         4.2308,
                         4.7436,
                         5.0000,]))

        y2 =np.vstack(np.array([

                        -0.9500,
                        -0.4372,
                        -0.3090,
                        -0.1949,
                        -0.1808,
                         0.0115,
                         0.1897,
                         0.2538,
                         0.3179,
                         0.4462,
                         0.4603,
                         0.5885,
                         0.6526,
                         1.1513,
                         1.1654,
                         1.2295,
                         1.3436,
                         1.3577,
                         1.5359,
                         1.6000,]))
        x11 =np.vstack(np.array([
                        -5.0000,
                        -5.0000,
                        -4.4872,
                        -4.4872,
                        -3.4615,
                        -3.4615,
                        -3.2051,
                        -3.2051,
                        -2.9487,
                        -2.9487,
                        -2.6923,
                        -2.6923,
                        -2.4359,
                        -2.4359,
                        -1.6667,
                        -1.6667,
                        -0.8974,
                        -0.8974,
                        -0.1282,
                        -0.1282,
                         0.1282,
                         0.6410,
                         0.8974,
                         1.4103,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795,
                         2.1795, ]))
        y11 =np.vstack(np.array([
                         0.4668,
                         1.5950,
                         0.7331,
                         1.6777,
                         0.8907,
                         1.9397,
                         1.0684,
                         1.8230,
                         0.0149,
                         1.0618,
                         0.1085,
                         1.2607,
                         0.2091,
                         1.1806,
                         0.4664,
                         1.6596,
                         0.7566,
                         1.7152,
                         1.0646,
                         1.7484,
                         0.8681,
                         1.0385,
                         1.2429,
                         1.3841,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,
                         1.7861,]))
        x22 =np.vstack(np.array([
                            -1.1538,
                            -1.1538,
                            -0.6410,
                            -0.6410,
                            -0.3846,
                            -0.3846,
                            -0.1282,
                            -0.1282,
                             0.3846,
                             0.3846,
                             0.6410,
                             0.6410,
                             1.1538,
                             1.1538,
                             1.4103,
                             1.4103,
                             3.2051,
                             3.2051,
                             3.2051,
                             3.4615,
                             3.4615,
                             3.4615,
                             3.7179,
                             3.7179,
                             3.7179,
                             3.9744,
                             3.9744,
                             3.9744,
                             4.2308,
                             4.2308,
                             4.2308,
                             4.7436,
                             4.7436,
                             4.7436,
                             5.0000,
                             5.0000,
                             5.0000,
                             5.0000,]))
        y22 = np.vstack(np.array([
                             0.0065,
                            -0.4440,
                             0.0993,
                            -0.3787,
                             0.2151,
                            -0.4539,
                             0.2596,
                            -0.2942,
                             0.2601,
                            -0.3071,
                             0.3095,
                            -0.2253,
                             0.3335,
                            -0.2139,
                             0.3966,
                            -0.1142,
                             1.0163,
                             0.2582,
                            -0.3984,
                             0.9459,
                             0.1728,
                            -0.3273,
                             0.7559,
                             0.2247,
                            -0.3612,
                             1.0864,
                             0.3206,
                            -0.2617,
                             0.7132,
                             0.3265,
                            -0.2551,
                             0.8570,
                             0.5128,
                            -0.1360,
                             0.9397,
                             0.5333,
                            -0.0796,
                            -0.5919,]))

    return x1, y1, x2, y2, x11, y11, x22, y22
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

def grafico_xai(num_amostras):
    NumberofHiddenNeurons = 100
    #grafico_xai_inter('sigmoid', NumberofHiddenNeurons, num_amostras)
    grafico_xai_inter('linear', NumberofHiddenNeurons, num_amostras)
    #grafico_xai_inter('radbas', NumberofHiddenNeurons, num_amostras)
    #grafico_xai_inter('sine', NumberofHiddenNeurons,num_amostras)
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
        TVP = np.vstack((benignTest, malignTest))
    
    print(NumberofInputNeurons)
    breakpoint()
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

    # Set benign values to -1
    T[1, :benignInput.shape[0]] = -1
    T[0, benignInput.shape[0]:] = -1
    breakpoint()
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
            TVP = TVP.T

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