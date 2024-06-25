from math import *
import numpy as np
import pandas as pd
from scipy import stats
from plotly.subplots import make_subplots

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
    pass
