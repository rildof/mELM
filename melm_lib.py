from math import *
from random import *
from time import process_time
import sys
import math
import time
import struct
import sys,string
import argparse
import numpy as np
import pandas as pd
import plotly.express as px

def process_uci_dataset(path):
	with open(path, 'r') as dataset:
		d = dataset.read()
		d = d.split("\n")
		d = [i.split(':') for i in d]
		x = []
		for row in d:
			y = []
			for cell in row:
				cell = cell.split(' ')[0]
				y.append(cell)
			x.append(y)

		df = pd.DataFrame(x).T
		df = df.drop(df.columns[[-1]], axis=1)
		df = df.T
		return df.astype(np.float64)
def processCSV_1(Data_File):
	data = pd.read_csv(Data_File, sep=' ', decimal=".", header=None)
	for ii in reversed(range(np.size(data, 1))):
		if np.isnan(data.loc[:, ii]).all():
			data.drop(data.columns[ii], axis=1, inplace=True)

	return data

def processCSV(Data_File):
	data = pd.read_csv(Data_File, sep=';', decimal=".", header=None)
	data = data.drop(0, axis=0).reset_index(drop=True).astype(np.float64)
	for ii in reversed(range(np.size(data, 1))):
		if np.isnan(data.loc[:, ii]).all():
			data.drop(data.columns[ii], axis=1, inplace=True)

	cols = list(data.columns.values)
	cols[11], cols[0] = cols[0], cols[11]
	data = data[cols]
	data.columns = range(data.columns.size)
	return data

def MakeTrainTest(dSet, percentTraining):
        #shuffle dataset
    dSet = dSet.sample(frac=1, random_state=349).reset_index(drop=True)
    splitPoint = int(len(dSet)*percentTraining)
    train_data, test_data = dSet[:splitPoint], dSet[splitPoint:]
    test_data = test_data.reset_index(drop=True)
    return [train_data, test_data]


"""all datas must be organized in the way of a dict:
Example: 2 graphs, with two lines each
datas = [ [T,Y], [TVT,TY] ]

if each has label:
[ [(T,'label1'),(Y,'label2')], [(TVT,'label3'),(TY,'label4)] ]
"""
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
