from math import *
import numpy as np
import pandas as pd
from scipy import stats
from plotly.subplots import make_subplots
from collections import Counter
import plotly.graph_objects as go


def graph(datas, titles, general_title):
    assert len(datas) == len(titles)
    traces = []
    for data in datas:
        trace = []
        for index, plot in enumerate(data):
            if index == 0:
                color = "#3D3B8E"
            elif index == 1:
                color = "#FFC53A"
            elif index == 2:
                color = "#F75C03"
            trace.append(go.Scatter(
                x=list(range(len(plot[0]))),
                y=plot[0],
                mode='markers',
                marker=dict(
                    color=color
                ),
                name=plot[1]
            ))
        traces.append(trace)
    numberOfGraphs = len(traces)
    maxRows = ceil(numberOfGraphs / 2.0)
    maxCols = 2
    fig = make_subplots(cols=2, rows=maxRows, subplot_titles=titles, vertical_spacing=0.05)
    row = 1
    plot_index = 1
    for index, plot in enumerate(traces):
        for indice, trace in enumerate(plot):
            fig.add_trace(trace, col=plot_index, row=row)
        plot_index += 1
        if plot_index > maxCols:
            row += 1
            plot_index = 1

    fig['layout'].update(height=600 * maxRows, width=1400, title=general_title, xaxis=dict(
        tickangle=-90
    ))
    fig.update_traces()
    fig.show()


def graph_classification(datas, titles, general_title):
    assert len(datas) == len(titles)
    traces = []
    for data in datas:
        assert len(data) == 2
        trace = []
        comparison = ["blue" if i == j else "red" for i,j in zip(data[0][0],data[1][0])]
        yaxis = ["correct" if i == j else "wrong" for i,j in zip(data[0][0],data[1][0])]
        trace.append(go.Scatter(
            x=list(range(len(data[0][0]))),
            y=yaxis,
            mode='markers',
            marker=dict(
                color=comparison,
                size=8
            ),
            name=data[0][1]
        ))
        traces.append(trace)
    numberOfGraphs = len(traces)
    maxRows = ceil(numberOfGraphs / 2.0)
    maxCols = 1
    fig = make_subplots(cols=maxCols, rows=maxRows, subplot_titles=titles, vertical_spacing=0.05)
    row = 1
    plot_index = 1
    for index, plot in enumerate(traces):
        for indice, trace in enumerate(plot):
            fig.add_trace(trace, col=plot_index, row=row)
        plot_index += 1
        if plot_index > maxCols:
            row += 1
            plot_index = 1

    fig['layout'].update(height=600 * maxRows, width=1500, title=general_title, xaxis=dict(
        tickangle=-90
    ))
    fig.update_traces()
    fig.show()


def graph_Train_Test(lst_train, lst_test, title):
    traces_train, traces_test = [], []
    for array in lst_train:
        traces_train.append(go.Scatter(
            x=list(range(len(array))),
            y=array,
            mode='markers',
            marker=dict(
                color=f'rgb({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)})'
            )
        ))

    for array in lst_test:
        traces_test.append(go.Scatter(
            x=list(range(len(array))),
            y=array,
            mode='markers',
            marker=dict(
                color=f'rgb({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)})'
            )
        ))
    fig = make_subplots(cols=2, rows=1, subplot_titles=("Training", "Testing"))
    for trace in traces_train:
        fig.add_trace(trace, col=1, row=1)
    for trace in traces_test:
        fig.add_trace(trace, col=2, row=1)

    fig['layout'].update(height=720, width=1280, title=title, xaxis=dict(
        tickangle=-90
    ))
    fig.update_traces()
    fig.show()


def remove_uniform_columns(df):
    df = df.drop([df.columns[0], df.columns[-1]], axis=1)
    df = df.drop([0])
    df = df[[c for c in list(df) if len(df[c].unique()) > 1]]
    df.columns = range(df.columns.size)
    return df


def get_duplicate_rows(a, b):
    df = pd.concat([a, b])
    return df[df.duplicated(keep='last')]


def MakeTrainTest(dSet, percentTraining):
    dSet = dSet.sample(frac=1, random_state=349).reset_index(drop=True)
    splitPoint = int(len(dSet) * percentTraining)
    train_data, test_data = dSet[:splitPoint], dSet[splitPoint:]
    test_data = test_data.reset_index(drop=True)
    train_data = train_data.astype('float64')
    test_data = test_data.astype('float64')
    return [train_data, test_data]


def ProcessCSV(Benign_address, Malign_address, pvalue=0.05):
    malign = pd.read_csv(Malign_address, sep=';', decimal=".", header=None, low_memory=False)
    malign = malign.drop([malign.columns[0], malign.columns[-1]], axis=1)
    malign = malign.drop([0])
    malign = malign.astype('float64')
    malign.insert(0, 0, np.ones(len(malign)))

    benign = pd.read_csv(Benign_address, sep=';', decimal=".", header=None, low_memory=False)
    benign = benign.drop([benign.columns[0], benign.columns[-1]], axis=1)
    benign = benign.drop([0])
    benign = benign.astype('float64')
    benign.insert(0, 0, np.zeros(len(benign)))

    dSet = pd.concat([malign, benign], ignore_index=True)
    dSet = remove_uniform_columns(dSet)

    corr_df = pd.DataFrame(columns=['r', 'p-value'])

    for col in dSet:
        if pd.api.types.is_numeric_dtype(dSet[col]):
            r, p = stats.pearsonr(dSet[0], dSet[col])
            corr_df.loc[col] = [round(r, 3), round(p, 3)]

    condicao = corr_df['p-value'] < pvalue
    indices = corr_df[condicao]

    dSet = dSet[list(indices.index)]
    return dSet.reset_index(drop=True)


def ProcessCSV_2(Benign_address, Malign_address, pvalue=1):
    malign = pd.read_csv(Malign_address, sep=',', decimal=".", header=None, low_memory=False)
    malign = malign.drop([malign.columns[-1]], axis=1)
    malign = malign.drop([0])
    malign = malign.astype('float64')
    malign.insert(0, 0, np.ones(len(malign)), allow_duplicates=True)

    benign = pd.read_csv(Benign_address, sep=',', decimal=".", header=None, low_memory=False)
    benign = benign.drop([benign.columns[-1]], axis=1)
    benign = benign.drop([0])
    benign = benign.astype('float64')
    benign.insert(0, 0, np.zeros(len(benign)), allow_duplicates=True)

    dSet = pd.concat([malign, benign], ignore_index=True)
    dSet = remove_uniform_columns(dSet)

    return dSet.reset_index(drop=True)


def ProcessCSV_matlab(Benign_address, Malign_address):
    malign = pd.read_csv(Malign_address, sep=';', decimal=".", header=None, low_memory=False)
    malign = malign.astype('float64')

    benign = pd.read_csv(Benign_address, sep=';', decimal=".", header=None, low_memory=False)
    benign = benign.astype('float64')

    dSet = pd.concat([malign, benign], ignore_index=True)

    return dSet.reset_index(drop=True)

def get_all_modes(matrix):
    # function that gets all modes from each column of a matrix
    if type(matrix) == pd.DataFrame:
        matrix = matrix.T.values.tolist()
        modes = []
        for c in matrix:
            x = Counter(c).most_common()
            x = [round(c[0],6) for c in x]
            modes.append(x)
        return modes

def get_modes(matrix, num_of_modes):
    if type(matrix) == np.ndarray:
        modes = []
        matrix = matrix.T

        for c in matrix:
            column_modes = []
            for i in range(num_of_modes):
                mode = (stats.mode(c, axis=0, keepdims=True, nan_policy='omit')[0])[0]
                column_modes.append(mode)
                for value in c:
                    if value == mode:
                        c = np.delete(c, np.where(c == value))
            modes.append(column_modes)
        return modes

    elif type(matrix) == pd.DataFrame:
        modes = []
        matrix = matrix.T.astype('float64')
        for c in matrix.iterrows():
            c = c[1]
            column_modes = []
            for i in range(num_of_modes):
                if c.isnull().all():
                    if i == 0:
                        mode = np.nan
                    else:
                        mode = column_modes[-1]
                else:
                    mode = c.mode()[0]
                column_modes.append(mode)
                for index, value in enumerate(c):
                    if value == mode:
                        c[index] = None
            modes.append(column_modes)
        return modes

def get_weights(self, train_data:pd.DataFrame, classes):
    classesValues = list()
    for index, classLabel in enumerate(reversed(classes)):
        classesValues.append(train_data.loc[train_data[0] == classLabel].reset_index(drop=True))
        classesValues[index].drop(classesValues[index].columns[0], axis=1, inplace=True)
    
    train_data.drop(train_data.columns[0], axis=1, inplace=True)

    general_minimum = train_data.min(axis=None) 
    minimos = [c.min(axis=0) for c in classesValues]
    breakpoint()
    numberOfInputNeurons = classesValues[0].shape[1]
    
    def get_max(classValues:pd.DataFrame, general_minimum:float) -> list[list, float]:
        inputMax = float
        vectorInput = np.ndarray(classValues.shape[1])

        for index in range(classValues.shape[1] - 1, -1, -1):
            column = classValues.T.values[index]
            #TODO: Implementar is_atomic (pra quando a coluna for atômica ou não)
            if self.is_saturated(index, 1):
                vectorInput[index] = general_minimum
                continue
            vectorInput[index] = round(stats.mode(column)[0], 6)

        inputMax = max(vectorInput)
        return vectorInput, inputMax

    def pesos_xai_mode_auxiliar(self, vectorInput, inputMax, columnIndex, classIndex, count_vector):
        if self.iteration == 0:
            return vectorInput, inputMax, columnIndex, count_vector
        modes = get_all_modes(train_data)
        while True:
            #Verifica se o valor máximo, entre todos os neurônios da
            #classe, já está contido no referido neurônio de entrada.
            if inputMax in np.array(self.weights).T[columnIndex]:
                print('ja presente na coluna')
                if self.is_saturated(columnIndex, classIndex):
                    print('ja ta saturado')
                    vectorInput[columnIndex] = general_minimum
                    inputMax = max(vectorInput)
                    #atualiza direto no inputWeight
                    return vectorInput, inputMax, columnIndex, count_vector
                
                count_vector[columnIndex] = count_vector[columnIndex] + 1
                breakpoint()
                if count_vector[columnIndex] >= len(modes[columnIndex]):
                    print('acabou de saturar')
                    self.saturated_neurons[classIndex][columnIndex] = 1
                    vectorInput[columnIndex] = general_minimum
                    inputMax = max(vectorInput)

                    if sum(self.saturated_neurons[classIndex]) >= numberOfInputNeurons:
                        print('saturou toda a classe')
                        columnIndex = 0
                        vectorInput = []
                        return vectorInput, inputMax, columnIndex, count_vector
                    
                    print('saturou parcial')
                    columnIndex = numberOfInputNeurons
                    return vectorInput, inputMax, columnIndex, count_vector
                print('substituicao')
                print('colunaModificada:', columnIndex)
                print(f'valor antigo: {vectorInput[columnIndex]} novo valor: {modes[columnIndex][int(count_vector[columnIndex])]}')
                vectorInput[columnIndex] = modes[columnIndex][int(count_vector[columnIndex])]
                inputMax = max(vectorInput)
                columnIndex = numberOfInputNeurons
                #breakpoint()
                #print(vectorInput)
                return vectorInput, inputMax, columnIndex, count_vector

            else: # A amostra é inédita
                break

        return vectorInput, inputMax, columnIndex, count_vector

    classesWeights = list()
    for index, classe in enumerate(classesValues): #TODO: Implementar o paralelismo # Para cada classe de valores(labels)
        if sum(self.saturated_neurons[index]) == len(self.saturated_neurons[index]): break

        vectorInput, inputMax = get_max(classe, general_minimum)
        ii = classe.shape[1] - 1
        count_vector = np.zeros(numberOfInputNeurons)
        while ii >= 0:
            #if not self.is_saturated(ii, index):
            [vectorInput, inputMax, ii, count_vector] =(
                pesos_xai_mode_auxiliar(self, vectorInput, inputMax, ii, index, count_vector))
            #else:
            #    print('ii saturado:', ii)
            ii -= 1
        classesWeights.append(vectorInput)
    print('pesos calculados')
    return classesWeights
def matprint(mat, fmt=".4f"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
#========================================================================
def switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,P):

    if (ActivationFunction == 'sig') or (ActivationFunction == 'sigmoid'):
        H = sig_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'sin') or (ActivationFunction == 'sine'):
        H = sin_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'hardlim'):
        H = hardlim_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'tribas'):
        H = tribas_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'radbas'):
        H = radbas_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'erosion'):
        H = erosion(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'dilation'):
        H = dilation(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'fuzzy-erosion') or (ActivationFunction == 'fuzzy_erosion'):
        H = fuzzy_erosion(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'fuzzy-dilation') or (ActivationFunction == 'fuzzy_dilation'):
        H = fuzzy_dilation(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'bitwise-erosion') or (ActivationFunction == 'bitwise_erosion'):
        H = bitwise_erosion(InputWeight, BiasofHiddenNeurons, P)
    elif (ActivationFunction == 'bitwise-dilation') or (ActivationFunction == 'bitwise_dilation'):
        H = bitwise_dilation(InputWeight, BiasofHiddenNeurons, P)
    else: #'linear'
        H = linear_kernel(InputWeight, BiasofHiddenNeurons, P)
    return H
#========================================================================
def sig_kernel(w1, b1, samples):
    #%%%%%%%% Sigmoid
    tempH = np.dot(w1, samples) + b1
    H = 1 / (1 + np.exp(-tempH))
    return H
#========================================================================
def sin_kernel(w1, b1, samples):
    #%%%%%%%% Sine
    tempH = np.dot(w1, samples) + b1
    H = np.sin(tempH)
    return H
#========================================================================
def hardlim_kernel(w1, b1, samples):
    #%%%%%%%% Hard Limit
    #hardlim(n)	= 1 if n ≥ 0
    #		= 0 otherwise
    tempH = np.dot(w1, samples) + b1
    H = tempH
    for ii in range(np.size(tempH,0)):
        for jj in range(np.size(tempH,1)):
            if tempH[ii][jj] >= 0:
                H[ii][jj] = 1
            else:
                H[ii][jj] = 0
    return H
#========================================================================
def tribas_kernel(w1, b1, samples):
    #%%%%%%%% Triangular basis function
    #a = tribas(n) 	= 1 - abs(n), if -1 <= n <= 1
        #	 	= 0, otherwise
    tempH = np.dot(w1, samples) + b1
    H = tempH
    for ii in range(np.size(tempH,0)):
        for jj in range(np.size(tempH,1)):
            if (tempH[ii][jj] >= -1) and (tempH[ii][jj] <= 1):
                H[ii][jj] = 1 - abs(tempH[ii][jj])
            else:
                H[ii][jj] = 0
    return H
#========================================================================
def radbas_kernel(w1, b1, samples):
    #%%%%%%%% Radial basis function
    #radbas(n) = exp(-n^2)
    tempH = np.dot(w1, samples) + b1
    H = np.exp(-np.power(tempH, 2))
    return H
#========================================================================
def linear_kernel(w1, b1, samples):
    H = np.dot(w1, samples) + b1
    return H
#========================================================================
def erosion(w1, b1, samples):

    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1))

    for s_index in range(np.size(samples,1)):
        ss = samples.loc[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = max(ss.loc[j], 1-w1[i][j])
            H[i][s_index] = min(x)+b1[i][0]

    return H
#========================================================================
def dilation(w1, b1, samples):
    
    if True:
        if type(samples) == np.ndarray:
            samples = pd.DataFrame(samples, columns=range(samples.shape[1]))
        H = np.zeros((np.size(w1,0), np.size(samples,1)))
        x = np.zeros(np.size(w1,1))

        for s_index in range(np.size(samples,1)):
            ss = samples.loc[:,s_index]
            for i in range(np.size(w1,0)):
                for j in range(np.size(w1,1)):
                    x[j] = min(ss.loc[j], w1[i][j])
                H[i][s_index] = max(x)+ b1[i][0]
    else:
        NumberofHiddenNeurons = w1.shape[0]
        NumberofTrainingData = samples.shape[1]
        H = np.ones((NumberofHiddenNeurons, NumberofTrainingData))
        NumberofAtributos = w1.shape[1]
        if True:   
            for i in range(NumberofHiddenNeurons):
                for j in range(NumberofTrainingData):
                    result = min(w1[i, 0], samples[0, j])
                    
                    for k in range(1, NumberofAtributos):
                        temp = min(w1[i, k], samples[k, j])
                        
                        if temp > result:
                            result = temp
                            
                    H[i, j] = result
    return H
#========================================================================
def fuzzy_erosion(w1, b1, samples):

    tempH = np.dot(w1, samples) + b1
    H = np.ones((np.size(w1,0), np.size(samples,1)))

    for s_index in range(np.size(samples,1)):
        #ss = samples.loc[:,s_index]
        for i in range(np.size(w1,0)):
            H[i][s_index] = 1- tempH[i][s_index]
    return H
#========================================================================
def fuzzy_dilation(w1, b1, samples):

    tempH = np.dot(w1, samples) + b1
    H = np.ones((np.size(w1,0), np.size(samples,1)))

    for s_index in range(np.size(samples,1)):
        ss = samples.loc[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                H[i][s_index] = H[i][s_index] * (1 - tempH[i][j])

    H = 1 - H
    return H
#========================================================================
def bitwise_erosion(w1, b1, samples):

    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1),dtype=bytearray)

    for s_index in range(np.size(samples,1)):
        ss = samples.loc[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = bytes_or(ss.loc[j], 1-w1[i][j])
            result = x[0]
            for j in range(1, np.size(w1,1)):
                result = bytes_and(result, x[j])
            temp = struct.unpack('d', result)[0]
            if math.isnan(temp): temp = 0.0
            H[i][s_index] = temp + b1[i][0]
    return H
#========================================================================
def bitwise_dilation(w1, b1, samples):

    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1),dtype=bytearray)

    for s_index in range(np.size(samples,1)):
        ss = samples.loc[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = bytes_and(ss.loc[j], w1[i][j])
            result = x[0]
            for j in range(1, np.size(w1,1)):
                result = bytes_or(result, x[j])
            temp = struct.unpack('d', result)[0]
            if math.isnan(temp): temp = 0.0
            H[i][s_index] = temp + b1[i][0]
    return H

#========================================================================
def bytes_and(a, b) :
    a1 = bytearray(a)
    b1 = bytearray(b)
    c = bytearray(len(a1))

    for i in range(len(a1)):
        c[i] = a1[i] & b1[i]
    return c

#========================================================================
def bytes_or(a, b) :
    a1 = bytearray(a)
    b1 = bytearray(b)
    c = bytearray(len(a1))

    for i in range(len(a1)):
        c[i] = a1[i] | b1[i]
    return c

if __name__ == '__main__':
    pass
