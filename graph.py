import os 
import numpy as np
from itertools import product
from melm_lib import *
from scipy.linalg import pinv
import plotly.graph_objects as go
import os

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

    # Calculate accuracy
    accuracy = 1 - (classificacoesErradas / NumberofTestingData)
    accuracy = round(accuracy * 100, 2)
    
    return accuracy

def avaliarRede(TY, TVP):
    # Get the index of the maximum output (winner neuron) for each pattern
    nodoVencedorRede = np.argmax(TY, axis=0)

    # Initialize empty lists to hold x and y coordinates
    x1, y1, x2, y2 = [], [], [], []

    # Iterate over each pattern
    for padrao in range(TVP.shape[1]):
        if nodoVencedorRede[padrao] == 1:
            x1.append(TVP[0, padrao])
            y1.append(TVP[1, padrao])
        else:
            x2.append(TVP[0, padrao])
            y2.append(TVP[1, padrao])

    # Convert lists to numpy arrays
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

def distribuicao(dist):
    # Definindo o número de amostras
    num_amostras = 40
    
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
    acc = avaliarRedeTreino(Y, NumberofTrainingData, T)

    # Calculate the hidden layer output matrix for the test data (H_test)
    H_test = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, TVP)
    TY = (H_test.T @ OutputWeight).T
    del H_test  # Clear H_test to save memory

    # Evaluate the network for testing
    xx1, yy1, xx2, yy2 = avaliarRede(TY, TVP)

    breakpoint()
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
    acc = avaliarRedeTreino(Y, NumberofTrainingData, T)
    
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

def grafico_xai():
    NumberofHiddenNeurons = 100
    #grafico_xai_inter('sigmoid', NumberofHiddenNeurons)
    grafico_xai_inter('linear', NumberofHiddenNeurons)
    #grafico_xai_inter('radbas', NumberofHiddenNeurons)
    #grafico_xai_inter('sine', NumberofHiddenNeurons)

def grafico_xai_inter(kernel, NumberofHiddenNeurons):

    [x1, y1, x2, y2, x11, y11, x22, y22] = distribuicao(kernel)

    iteracao = 1
    NumberofInputNeurons = 2

    # Stack arrays vertically (equivalent to MATLAB's vertcat)
    tempa = np.vstack((x1, x11))
    tempb = np.vstack((y1, y11))
    # Stack arrays horizontally (equivalent to MATLAB's horzcat)
    entrada_benigno = np.hstack((tempa, tempb))

    tempa = np.vstack((x2, x22))
    tempb = np.vstack((y2, y22))
    entrada_maligno = np.hstack((tempa, tempb))

    # Stack benigno and maligno entries vertically and transpose (equivalent to vertcat and transpose)
    P = np.vstack((entrada_benigno, entrada_maligno)).T
    
    #Pesos para as execuções padrão
    InputWeight, BiasofHiddenNeurons = pesos(iteracao, NumberofHiddenNeurons, NumberofInputNeurons)

    #--------------------------------------------------------------------RILDO
    entrada_benigno_Rildo = entrada_benigno  
    entrada_maligno_Rildo = entrada_maligno

    #insert value of 1 in the first column of the benigno matrix
    entrada_benigno_Rildo = np.insert(entrada_benigno_Rildo, 0, 1, axis=1)
    #insert value of 2 in the first column of the maligno matrix
    entrada_maligno_Rildo = np.insert(entrada_maligno_Rildo, 0, 2, axis=1)

    conjuntoTreinamento = np.vstack((entrada_benigno_Rildo, entrada_maligno_Rildo))

    #InputWeight_xai_Rildo, InputWeightClass = pesos_xai_rildo(
    #    iteracao, NumberofInputNeurons, conjuntoTreinamento
    #)
    #---------------------------------------------------------------------------- RILDO



    # Initialize T with ones
    T = np.ones((2, P.shape[1]))

    # Set values in T based on the condition
    T[1, :entrada_benigno.shape[0]] = -1
    T[0, entrada_benigno.shape[0]:] = -1

    # Calculate minimum and maximum values for P
    minP1 = np.min(P[0, :])
    maxP1 = np.max(P[0, :])
    minP2 = np.min(P[1, :])
    maxP2 = np.max(P[1, :])

    # Create vectors using linspace
    vetora = np.linspace(minP1, maxP1, 160)
    vetorb = np.linspace(minP2, maxP2, 80)

    # Create the combinatorial matrix similar to combvec in MATLAB
    TVP = np.array(list(product(vetora, vetorb))).T

    NumberofTestingData = TVP.shape[1]
    NumberofTrainingData = P.shape[1]

    # ELM Clássica
    # grafico_auxiliar('PINHEIRO','dilatacao_classica', kernel, NumberofHiddenNeurons,
    #                      NumberofTrainingData, NumberofTestingData,
    #                      InputWeight, P, T, TVP,
    #                      BiasofHiddenNeurons,
    #                      entrada_benigno, entrada_maligno)
    
    # grafico_auxiliar('PINHEIRO','erosao_classica', kernel, NumberofHiddenNeurons,
    #                     NumberofTrainingData,NumberofTestingData,
    #                     InputWeight, P, T, TVP,
    #                     BiasofHiddenNeurons,
    #                     entrada_benigno, entrada_maligno)
    #ELM sem camada escondida
    grafico_nohidden_auxiliar('HUANG',kernel, kernel, NumberofHiddenNeurons,
                            NumberofTrainingData,NumberofTestingData, 
                            InputWeight, P, T, TVP,
                            BiasofHiddenNeurons,
                            entrada_benigno, entrada_maligno)

    # grafico_nohidden_auxiliar('HUANG', kernel, 'ero_fuzzy', NumberofHiddenNeurons,
    #                             NumberofTrainingData,NumberofTestingData, 
    #                             InputWeight, P, T, TVP,
    #                             BiasofHiddenNeurons,
    #                             entrada_benigno, entrada_maligno) 
    #ELM XAI
    #ELM XAI Rildo
    

x = grafico_xai()