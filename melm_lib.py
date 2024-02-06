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

def processCSV(Data_File):
	data = pd.read_csv(Data_File, sep=' ', decimal=".", header=None)
	for ii in reversed(range(np.size(data, 1))):
		if np.isnan(data.loc[:, ii]).all():
			data.drop(data.columns[ii], axis=1, inplace=True)

	return data

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
