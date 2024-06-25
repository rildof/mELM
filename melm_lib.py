from math import *
import numpy as np
import pandas as pd
from scipy import stats

def graph(datas, titles, general_title):
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	assert len(datas) == len(titles)
	traces = []
	for data in datas: #Para cada gráfico que queira fazer
		trace = []
		for index,plot in enumerate(data): #Para cada conjunto de dados desse gráfico
			if index == 0 : color = "#3D3B8E"
			if index == 1 : color = "#FFC53A"
			if index == 2 : color = "#F75C03"
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
	maxRows=ceil(numberOfGraphs/2.0)
	maxCols = 2
	fig = make_subplots(cols=2, rows=maxRows, subplot_titles=titles, vertical_spacing=0.05)
	row = 1
	plot_index = 1
	for index, plot in enumerate(traces):
		for indice, trace in enumerate(plot):
			fig.add_trace(trace, col=plot_index, row=row)
		plot_index = plot_index + 1
		if plot_index > maxCols:
			row = row + 1
			plot_index = 1

	fig['layout'].update(height=600*maxRows, width=1400, title=general_title, xaxis=dict(
		tickangle=-90
	))
	fig.update_traces()
	fig.show()


def graph_Train_Test(lst_train, lst_test, title):
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	traces_train, traces_test = [], []
	for array in lst_train:
		traces_train.append(go.Scatter(
			x=list(range(len(array))),
			y=array,
			mode='markers',
			marker=dict(
				color=f'rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})'
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
	#fig = make_subplots(specs=[[{"secondary_y": True}]])
	fig = make_subplots(cols=2, rows=1, subplot_titles=("Training", "Testing"))
	for trace in traces_train: fig.add_trace(trace, col=1, row=1);
	for trace in traces_test: fig.add_trace(trace, col=2, row=1)

	fig['layout'].update(height=720, width=1280, title=title, xaxis=dict(
		tickangle=-90
	))
	fig.update_traces()
	fig.show()


def remove_uniform_columns(df):
        df = df.drop([df.columns[0], df.columns[-1]], axis=1)
        df = df.drop([0])
        # cols = df.select_dtypes([np.number]).columns
        # diff = df[cols].diff().abs().sum()
        # df = df.drop(diff[diff == 0].index, axis=1)
        df = df[[c for c
                         in list(df)
                         if len(df[c].unique()) > 1]]
        df.columns = range(df.columns.size)
        return df


def get_duplicate_rows(a,b):
    #a = pd.DataFrame([[1,2,3],[2,3,4],[4,5,6]])
    #b = pd.DataFrame([[1,2,3],[4,5,6],[6,7,8]])
    #print(a.iloc[0].equals(b.iloc[0]))
    df = pd.concat([a,b])
    return df[df.duplicated(keep = 'last')]


def MakeTrainTest(dSet, percentTraining):
        #shuffle dataset
    dSet = dSet.sample(frac=1, random_state=349).reset_index(drop=True)
    splitPoint = int(len(dSet)*percentTraining)
    train_data, test_data = dSet[:splitPoint], dSet[splitPoint:]
    test_data = test_data.reset_index(drop=True)
    return [train_data, test_data]


def ProcessCSV(Benign_address , Malign_address , pvalue=0.05):
    #This dataset is divided into malign and benign, so we need to merge them in a random way into a training array and a testing array
    #   @ - load malign and benign data
    malign = pd.read_csv(Malign_address, sep=';', decimal=".", header=None, low_memory= False)
        #This part is subjective to each dataset and should be changed if the dataset is changed
    malign = malign.drop([malign.columns[0], malign.columns[-1]], axis=1)
    malign = malign.drop([0])
    malign = malign.astype('float64')  # Convert string to int64
    #malign = malign.replace(0,-0.1)
    malign.insert(0,0, np.ones(len(malign))) #add a column of ones to the class dataset

    benign = pd.read_csv(Benign_address, sep=';', decimal=".", header=None, low_memory = False)
        #This part is subjective to each dataset and should be changed if the dataset is changed
    benign = benign.drop([benign.columns[0], benign.columns[-1]], axis=1) # remove first and last columns which is text and NaN respectively
    benign = benign.drop([0]) #remove first row which is only text
    benign = benign.astype('float64') # Convert string to int64
    #benign = benign.replace(0, -0.1)
    benign.insert(0,0, np.zeros(len(benign))) #add a column of zeros to the counter-class dataset
    #   @ - merge the two datasets, making a training dataset and a testing dataset
    dSet = pd.concat([malign, benign], ignore_index=True)
    dSet = remove_uniform_columns(dSet)

    corr_df = pd.DataFrame(columns=['r', 'p-value'])

    for col in dSet:  # Use this to loop through the DataFrame
        if pd.api.types.is_numeric_dtype(dSet[col]):  # Only calculate r, p-value for the numeric columns
            r, p = stats.pearsonr(dSet[0], dSet[
                col])  # .pearsonr() returns two values in a list, store them individually using this format
            # r and p-values are calculated in comparison with what it is(benign or malign)
            corr_df.loc[col] = [round(r, 3), round(p, 3)]  # Add the r & p for this col into the corr_df

    condicao = corr_df['p-value'] < pvalue
    indices = corr_df[condicao]

    dSet = dSet[list(indices.index)]
    return dSet.reset_index(drop=True)


def ProcessCSV_2(Benign_address , Malign_address , pvalue=1):
    #This dataset is divided into malign and benign, so we need to merge them in a random way into a training array and a testing array
    #   @ - load malign and benign data
    malign = pd.read_csv(Malign_address, sep=',', decimal=".", header=None, low_memory= False)
        #This part is subjective to each dataset and should be changed if the dataset is changed
    malign = malign.drop([malign.columns[-1]], axis=1)
    malign = malign.drop([0])
    malign = malign.astype('float64')  # Convert string to int64
    #malign = malign.replace(0,-0.1)
    malign.insert(0,0, np.ones(len(malign)) , allow_duplicates = True) #add a column of ones to the class dataset
    benign = pd.read_csv(Benign_address, sep=',', decimal=".", header=None, low_memory = False)
        #This part is subjective to each dataset and should be changed if the dataset is changed
    benign = benign.drop([benign.columns[-1]], axis=1) # remove first and last columns which is text and NaN respectively
    benign = benign.drop([0]) #remove first row which is only text
    benign = benign.astype('float64') # Convert string to int64
    #benign = benign.replace(0, -0.1)
    benign.insert(0,0, np.zeros(len(benign)), allow_duplicates= True) #add a column of zeros to the counter-class dataset
    #   @ - merge the two datasets, making a training dataset and a testing dataset
    dSet = pd.concat([malign, benign], ignore_index=True)
    dSet = remove_uniform_columns(dSet)

    #corr_df = pd.DataFrame(columns=['r', 'p-value'])

    #for col in dSet:  # Use this to loop through the DataFrame
    #    if pd.api.types.is_numeric_dtype(dSet[col]):  # Only calculate r, p-value for the numeric columns
    #        r, p = stats.pearsonr(dSet[0], dSet[
    #            col])  # .pearsonr() returns two values in a list, store them individually using this format
            # r and p-values are calculated in comparison with what it is(benign or malign)
    #        corr_df.loc[col] = [round(r, 3), round(p, 3)]  # Add the r & p for this col into the corr_df

    #condicao = corr_df['p-value'] < pvalue
    #indices = corr_df[condicao]

    #dSet = dSet[list(indices.index)]
    return dSet.reset_index(drop=True)


def ProcessCSV_matlab(Benign_address , Malign_address):
    #This dataset is divided into malign and benign, so we need to merge them in a random way into a training array and a testing array
    #   @ - load malign and benign data
    malign = pd.read_csv(Malign_address, sep=';', decimal=".", header=None, low_memory= False)
        #This part is subjective to each dataset and should be changed if the dataset is changed
    malign = malign.astype('float64')  # Convert string to float64
    
    benign = pd.read_csv(Benign_address, sep=';', decimal=".", header=None, low_memory = False)
    benign = benign.astype('float64') # Convert string to float64
    dSet = pd.concat([malign, benign], ignore_index=True)

    return dSet.reset_index(drop=True)

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


if __name__ == '__main__':
    #get_modes(np.array([[1,2,3],[1,2,3],[2,3,4]]), 2)
    #print(get_modes(pd.DataFrame([[1,2,3],[1,2,3],[2,3,4]]), 2))
    pass