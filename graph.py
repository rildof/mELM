import os 
import numpy as np
from itertools import product
from melm_lib import *
from scipy.linalg import pinv
import plotly.graph_objects as go
from scipy.stats import mode
import time
import copy
from sklearn.datasets import make_classification
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
           entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc):

    # Create scatter plot for xx1, yy1 (blue 'x') and xx2, yy2 (red 'o')
    fig = go.Figure()

    # Plot testing data points
    fig.add_trace(go.Scatter(x=xx1, y=yy1, mode='markers',
                             marker=dict(size=5, color='blue', symbol='x'),
                             name='Class 1'))
    
    fig.add_trace(go.Scatter(x=xx2, y=yy2, mode='markers',
                             marker=dict(size=5, color='red', symbol='circle'),
                             name='Class 2'))
    # Plot training data points for benign and malignant
    x1 = entrada_benigno[:, 0]
    y1 = entrada_benigno[:, 1]

    x2 = entrada_maligno[:, 0]
    y2 = entrada_maligno[:, 1]

    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers',
                             marker=dict(size=10, color='red', symbol='x'),
                             name='Benigno'))

    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers',
                             marker=dict(size=10, color='blue', symbol='circle'),
                             name='Maligno'))

    # Hide axis ticks and adjust layout
    fig.update_layout(
        showlegend=True,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        autosize=True,
        title=f'{author} - {ActivationFunction} Hidden Neurons: {NumberofHiddenNeurons}, Accuracy: {acc}%',
    )

    # Create folder if it doesn't exist
    #folder_path = f'ELM_XAI/{pasta}'
    #os.makedirs(folder_path, exist_ok=True)
    
    # Save the plot as a .bmp image
    #file_name = f'{author}_{ActivationFunction}_melm_hidden_neurons_{NumberofHiddenNeurons}_accuracy_{acc}.bmp'
    #file_path = os.path.join(folder_path, file_name)
    #fig.write_image(file_path)
    fig.show()

    #print(f'Plot saved to {file_path}')

def plotar_Rildo(author, pasta, ActivationFunction, NumberofHiddenNeurons,
                 InputWeight, InputWeightClass, 
                 entrada_benigno, entrada_maligno, xx1, yy1, xx2, yy2, acc):

    # Criação da figura
    fig = go.Figure()

    # Scatter plot para xx1, yy1 e xx2, yy2
    fig.add_trace(go.Scatter(
        x=xx1, y=yy1,
        mode='markers',
        marker=dict(symbol='x', size=5, color='blue'),
        name='Class 1'
    ))

    fig.add_trace(go.Scatter(
        x=xx2, y=yy2,
        mode='markers',
        marker=dict(symbol='circle', size=5, color='red'),
        name='Class 2'
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
        marker=dict(symbol='x', size=10, color='blue', line=dict(width=1.25)),
        name='Benigno'
    ))

    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='markers',
        marker=dict(symbol='circle', size=10, color='red', line=dict(width=1.25)),
        name='Maligno'
    ))

    # Adicionar textos sobre os pontos
    for i in range(len(x3)):
        fig.add_trace(go.Scatter(
            x=[x3[i]], y=[y3[i]],
            mode='text',
            text=[str(indice_linhas_1[i])],
            textposition='middle center',
            textfont=dict(size=30, color='blue')
        ))

    for i in range(len(x4)):
        fig.add_trace(go.Scatter(
            x=[x4[i]], y=[y4[i]],
            mode='text',
            text=[str(indice_linhas_2[i])],
            textposition='middle center',
            textfont=dict(size=30, color='red')
        ))

    # Ajuste da aparência dos eixos
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), autosize=True)

    # Salvar a figura como imagem
    #output_dir = f"ELM_XAI/{pasta}/"
    #os.makedirs(output_dir, exist_ok=True)
    #output_file = f"{output_dir}{author}_{ActivationFunction}_melm_hidden_neurons_{NumberofHiddenNeurons}_accuracy_{acc}_.png"
    #fig.write_image(output_file)

    # Exibir a figura
    fig.show()


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

# Função auxiliar (precisa ser implementada)
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
    H = kernel_matrix_rildo(ActivationFunction, InputWeight, BiasMatrix,  P)
    # Calculate output weights (beta_i)
    OutputWeight = np.linalg.pinv(H.T) @ T.T
    Y = (H.T @ OutputWeight).T
    
    end_time_train = time.time()
    TrainingTime = end_time_train - start_time_train
    
    # Calculate the output of testing input
    start_time_test = time.time()
    
    H_test = kernel_matrix_rildo(ActivationFunction, InputWeight, BiasMatrix,  TVP)
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
    num_amostras = 100
    
    if dist == 'sine':
        x1 = np.arange(-2 * np.pi, 2 * np.pi, 0.4)
        y1 = np.sin(x1)
        x2 = x1
        y2 = y1
        y1 = y1 + 0.2
        y2 = y2 - 0.2
        y1, x1 = adicionar_ruido_peq_neg(y1, x1, 2)
        
        y1a = y1 + 0.05
        y1 = np.hstack((y1a, y1))
        x1 = np.hstack((x1, x1))
        
        y2, x2 = adicionar_ruido_peq_pos(y2, x2, 2)
        y2a = y2 - 0.05
        y2b = y2 - 0.1
        y2c = y2 - 0.15
        y2 = np.hstack((y2a, y2, y2b, y2c))
        x2 = np.hstack((x2, x2, x2, x2))
        
        y1, x1, y2, x2 = sine_distr(y1, x1, y2, x2)

    elif dist == 'sigmoid':
        x1 = np.linspace(-10, 10, num_amostras)
        y1 = 1 / (1 + np.exp(-x1)) + 0.02 * np.random.randn(num_amostras)
        y1[y1 > 1] = 1
        y1[y1 < 0] = 0
        x2 = x1
        y2 = y1 - 0.1
        
        y1, x1, y2, x2 = sigmoid_distr(y1, x1, y2, x2)

    elif dist == 'radbas':
        centro = 0
        desvio_padrao = 1
        x1 = np.linspace(-5, 5, num_amostras)
        x2 = x1
        y1 = np.exp(-(x1 - centro) ** 2 / (2 * desvio_padrao ** 2)) + 0.15
        y2 = y1 - 0.15
        
        y1, x1 = adicionar_ruido_peq_pos(y1, x1, 1)
        y2, x2 = adicionar_ruido_peq_neg(y2, x2, 1)
        
        y1, x1, y2, x2 = radbas_distr(y1, x1, y2, x2)

    else:
        coeficiente_angular = 0.25
        intercepto = 0.55
        x1 = np.linspace(-5, 5, num_amostras)
        x2 = x1
        y1 = coeficiente_angular * x1 + intercepto
        y2 = y1
        y1 = y1 + 0.25
        y2 = y2 - 0.25
        y1, x1 = adicionar_ruido_peq_neg(y1, x1, 1)
        y2, x2 = adicionar_ruido_peq_pos(y2, x2, 1)
        y1, x1 = eliminar_partes(y1, x1, 0.5)
        y2, x2 = eliminar_partes(y2, x2, 0.5)
        
    y11, x11 = preencher_espacos_min(y1, x1)
    y22, x22 = preencher_espacos_max(y2, x2)

    mmin = min(len(y11), len(y22))
    mmax = max(len(y11), len(y22))

    tt = len(y11)
    for ii in range(mmin, mmax):
        if mmin == tt:
            y11 = np.append(y11, y11[mmin-1])
            x11 = np.append(x11, x11[mmin-1])
        else:
            y22 = np.append(y22, y22[mmin-1])
            x22 = np.append(x22, x22[mmin-1])

    x1 = dime(x1)
    y1 = dime(y1)
    x11 = dime(x11)
    y11 = dime(y11)
    x2 = dime(x2)
    y2 = dime(y2)
    x22 = dime(x22)
    y22 = dime(y22)
    
    return x1, y1, x2, y2, x11, y11, x22, y22

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

    NumberofHiddenNeurons = InputWeight.shape[0]
    BiasMatrix =  np.zeros((NumberofHiddenNeurons, 1))
    #----------------------------------------------------------------------

    H = kernel_matrix_rildo(ActivationFunction, InputWeight, BiasMatrix, P)
    
    OutputWeight = np.linalg.pinv(H.T) @ T.T
    Y = (H.T @ OutputWeight).T  
    del H
    
    acc, wrongIndexes = avaliarRedeTreino(Y, NumberofTrainingData, T)
    print(P[:,wrongIndexes].T)

    H_test = kernel_matrix_rildo(ActivationFunction, InputWeight, BiasMatrix, TVP)
    
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
    #grafico_xai_inter('sigmoid', NumberofHiddenNeurons)
    #grafico_xai_inter('linear', NumberofHiddenNeurons)
    #grafico_xai_inter('radbas', NumberofHiddenNeurons)
    grafico_xai_inter('sine', NumberofHiddenNeurons,num_amostras)
    #grafico_xai_inter('authoral', NumberofHiddenNeurons,num_amostras)
    


def grafico_xai_inter(kernel, NumberofHiddenNeurons,num_amostras):
    if kernel != "authoral":
        [x1, y1, x2, y2, x11, y11, x22, y22] = distribuicao(kernel,num_amostras)
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
        vetora = np.linspace(minP1, maxP1, int(num_amostras*3.2))
        vetorb = np.linspace(minP2, maxP2, int(num_amostras*1.6))
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
    
    breakpoint()
    NumberofTestingData = TVP.shape[1]
    NumberofTrainingData = P.shape[1]

    # ELM Clássica
    # grafico_auxiliar('PINHEIRO','dilatacao_classica', kernel, NumberofHiddenNeurons,
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
    

num_amostras =  70
x = grafico_xai(num_amostras)