import pandas as pd
import numpy as np
from melm_source import melm
def process_uci_dataset(path):
	with open(path, 'r') as dataset:
		d = dataset.read()
		d = d.split("\n")
		d = [i.split(':') for i in d]
		print(d)
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
		print(df)
		return df

path = r'datasets/uci/housing.txt'
process_uci_dataset(path)
