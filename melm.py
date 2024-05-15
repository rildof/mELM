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
from melm_source import melm
from melm_lib import processCSV, graph_Train_Test, MakeTrainTest, graph, process_uci_dataset


#========================================================================
class authoral_melm():

	def __init__(self, error_limit):
		self.weights = []
		self.Accuracies = []
		self.error_limit = error_limit
		self.inputOutput = []

	def getInputOutput(self):
		return self.inputOutput
	def getLastAccuracy(self):
		return self.Accuracies[-1]
	def getWeightsLength(self):
		return len(self.weights)

	def changeErrorLimit(self, error_limit):
		self.error_limit = error_limit
	def main(self,train_data, test_data,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,verbose, lastRun, plot):
				
		if ActivationFunction is None:
			ActivationFunction = 'linear'
		print('kernel ' + ActivationFunction)
		
		if nSeed is None:
			nSeed = 1
		else:
			nSeed = int(nSeed)
		seed(nSeed)
		np.random.seed(nSeed)

		Elm_Type = int(Elm_Type)
		NumberofHiddenNeurons = int(NumberofHiddenNeurons)
		if verbose: print ('Macro definition')
		#%%%%%%%%%%% Macro definition
		self.REGRESSION=0
		self.CLASSIFIER=1

		if verbose: print ('Load training dataset')
		#%%%%%%%%%%% Load training dataset

		T=np.transpose(train_data.loc[:,0])
		P=np.transpose(train_data.loc[:,1:np.size(train_data,1)])
		T = T.reset_index(drop=True)
		P = P.reset_index(drop=True)
		del(train_data)                                   #%   Release raw training data array
		
		if verbose: print ('Load testing dataset')
		#%%%%%%%%%%% Load testing dataset
		
		TVT=np.transpose(test_data.loc[:,0])
		TVP=np.transpose(test_data.loc[:,1:np.size(test_data,1)])
		TVT = TVT.reset_index(drop=True)
		TVP = TVP.reset_index(drop=True)
		del(test_data)                                    #%   Release raw testing data array

		NumberofTrainingData=np.size(P,1)
		NumberofTestingData=np.size(TVP,1)
		NumberofInputNeurons=np.size(P,0)


		if Elm_Type!=self.REGRESSION:
			if verbose: print ('Preprocessing the data of classification')
			#%%%%%%%%%%%% Preprocessing the data of classification
			sorted_target=np.sort(np.concatenate((T, TVT), axis=0))
			label = []                #%   Find and save in 'label' class label from training and testing data sets
			label.append(sorted_target[0])
			j=0
			for i in range(1, NumberofTrainingData+NumberofTestingData):
				if sorted_target[i] != label[j]:
					j=j+1
					label.append(sorted_target[i])
	
			number_class=j+1
			NumberofOutputNeurons=number_class
			if verbose: print ('Processing the targets of training')
			#%%%%%%%%%% Processing the targets of training
			temp_T=np.zeros((NumberofOutputNeurons, NumberofTrainingData))

			for i in range(0, NumberofTrainingData):
				for j in range(0, number_class):
					if label[j] == T[i]:
						break
				temp_T[j][i]=1
			T=temp_T*2-1

			if verbose: print ('Processing the targets of testing')
			#%%%%%%%%%% Processing the targets of testing
			temp_TV_T=np.zeros((NumberofOutputNeurons, NumberofTestingData))
			for i in range(0, NumberofTestingData):
				for j in range(0, number_class):
					if label[j] == TVT[i]:
						break
				temp_TV_T[j][i]=1
			TVT=temp_TV_T*2-1

		if verbose: print ('Calculate weights & biases')
		#%%%%%%%%%%% Calculate weights & biases
		start_time_train = process_time()

		if verbose: print ('Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons')
		#%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
		#print(P)
		if (ActivationFunction == 'erosion') or (ActivationFunction == 'dilation') or (ActivationFunction == 'fuzzy-erosion') or (ActivationFunction == 'fuzzy_erosion') or (ActivationFunction == 'fuzzy-dilation') or (ActivationFunction == 'fuzzy_dilation') or (ActivationFunction == 'bitwise-erosion') or (ActivationFunction == 'bitwise_erosion') or (ActivationFunction == 'bitwise-dilation') or (ActivationFunction == 'bitwise_dilation') :

			if lastRun:
				NumberofHiddenNeurons = len(self.weights) or 1
				InputWeight = np.array(self.weights)
				#InputWeight = np.reshape(InputWeight, (len(self.weights),1))
			else:
				self.weights.append(np.nanmean(P, axis=1))
				#InputWeight = np.array(np.reshape(np.nanmean(P),(1,1)))
				InputWeight = np.array([np.nanmean(P, axis=1)])
				#InputWeight = np.random.uniform(np.amin(np.amin(P)), np.amax(np.amax(P)),
				#								(NumberofHiddenNeurons, NumberofInputNeurons))
				#NumberofHiddenNeurons = 1
				#self.weights.append(InputWeight)

		else:
			InputWeight=np.random.rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;


		if lastRun:
			BiasofHiddenNeurons= np.random.rand(len(self.weights),1);
		else:
			BiasofHiddenNeurons = np.random.rand(1,len(InputWeight))


		if verbose: print ('Calculate hidden neuron output matrix H')
		#%%%%%%%%%%% Calculate hidden neuron output matrix H
		H = switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,P)
		
		if verbose: print ('Calculate output weights OutputWeight (beta_i)')
		#%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
		OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(T))
		
		end_time_train =  process_time()
		TrainingTime=end_time_train-start_time_train       # %   Calculate CPU time (seconds) spent for training ELM

		if verbose: print ('Calculate the training accuracy')
		#%%%%%%%%%%% Calculate the training accuracy
		Y = np.transpose(np.dot(np.transpose(H), OutputWeight))                     #%   Y: the actual output of the training data


		TrainingAccuracy = 0
		if Elm_Type == self.REGRESSION:        
			if verbose: print ('Calculate training accuracy (RMSE) for regression case')  
			#   Calculate training accuracy (RMSE) for regression case
			RMSE_tr = np.square(np.subtract(T,Y))
			TrainingAccuracy = RMSE_tr.mean()
			TrainingAccuracy = round(TrainingAccuracy, 6) 
			print('Training Accuracy: ' + str(TrainingAccuracy)+' ( ' + str(np.size(Y,0)) + ' samples) (regression)')
		del(H)
		
		if verbose: print ('Calculate the output of testing input')  
		start_time_test = process_time()
		#%%%%%%%%%%% Calculate the output of testing input
		tempH_test = switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,TVP)
		#del(TVP)

		TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))                     #%   Y: the actual output of the training data
		#TODO make plotly graph implementation
		#if plot: graph_Train_Test([T, Y], [TVT, TY], 'Authoral mELM')

		if plot: graph([ [(T,'Saída Desejada'),(Y,'Saída Obtida')],
						 [(TVT,'Saída Desejada'),(TY,'Saída Obtida')] ],
					   [ 'Treinamento', 'Teste'] ,
					   'authoral mELM')
		self.inputOutput = [T,Y,TVT,TY]
		end_time_test = process_time()
		TestingTime=end_time_test-start_time_test           #%   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

		TestingAccuracy = 0
		if Elm_Type == self.REGRESSION:          
			if verbose: print ('Calculate testing accuracy (RMSE) for regression case')  
			#   Calculate testing accuracy (RMSE) for regression case
			RMSE_ts = np.square(np.subtract(TVT, TY))
			TestingAccuracy = RMSE_ts.mean()
			TestingAccuracy = round(TestingAccuracy, 6) 
			print('Testing Accuracy: ' + str(TestingAccuracy)+' ( ' + str(np.size(TY,0)) + ' samples) (regression)')

		if Elm_Type == self.CLASSIFIER:
			if verbose: print ('Calculate training & testing classification accuracy')  
			#%%%%%%%%%% Calculate training & testing classification accuracy
			MissClassificationRate_Training=0
			MissClassificationRate_Testing=0

			label_index_expected = np.argmax(T, axis=0)   # Maxima along the second axis
			label_index_actual = np.argmax(Y, axis=0)   # Maxima along the second axis
									
			for i in range(0, np.size(label_index_expected,0)):
				if label_index_actual[i]!=label_index_expected[i]:
					MissClassificationRate_Training=MissClassificationRate_Training+1

			TrainingAccuracy=1-MissClassificationRate_Training/np.size(label_index_expected,0)
			TrainingAccuracy = round(TrainingAccuracy, 6) 
			print('Training Accuracy: ' + str(TrainingAccuracy*100)+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Training),'/',str(np.size(label_index_expected,0)),') (classification)')

			label_index_expected = np.argmax(TVT, axis=0)   # Maxima along the second axis
			label_index_actual = np.argmax(TY, axis=0)   # Maxima along the second axis

			for i in range(0, np.size(label_index_expected,0)):
				if label_index_actual[i]!=label_index_expected[i]:
					MissClassificationRate_Testing=MissClassificationRate_Testing+1

			TestingAccuracy=1-MissClassificationRate_Testing/np.size(label_index_expected,0)
			TestingAccuracy = round(TestingAccuracy, 6) 
			print('Testing Accuracy: ' + str(TestingAccuracy*100)+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Testing),'/',str(np.size(label_index_expected,0)),') (classification)')

			print('Training Time: ' + str(round(TrainingTime,6)) + ' seconds')
			print('Testing Time: ' + str(round(TestingTime,6)) + ' seconds')

		#	Get testing values that have a Root Mean Squared Error (RMSE) greater than the limit for next iteration
		self.Accuracies.append([TrainingAccuracy, TestingAccuracy])


		mean_error_tr = np.mean(RMSE_tr)
		std_tr = np.std(RMSE_tr)
		newdata_index = [RMSE_tr.index[i] for i in range(len(RMSE_tr)) if mean_error_tr - std_tr <= RMSE_tr[i] <= mean_error_tr + std_tr]
		new_train = pd.concat([T[newdata_index], np.transpose(P[newdata_index])] , axis=1)
		mean_error_ts = np.mean(RMSE_ts)
		std_ts = np.std(RMSE_ts)
		newdata_index = [RMSE_ts.index[i] for i in range(len(RMSE_ts)) if mean_error_ts - std_ts <= RMSE_ts[i] <= mean_error_ts + std_ts]
		new_test = pd.concat([TVT[newdata_index], np.transpose(TVP[newdata_index])], axis=1)
		return new_train, new_test


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

	H = np.zeros((np.size(w1,0), np.size(samples,1)))
	if w1.ndim > 1:
		num_columns = np.size(w1,1)
	else:
		num_columns = 1 
	x = np.zeros(num_columns)
	for s_index in range(np.size(samples,1)):
		ss = samples.loc[:,s_index]
		for i in range(np.size(w1,0)):
			for j in range(num_columns):
				#print(ss.loc[j], w1[i][j])
				#print(type(ss.loc[j]) , type(w1[i][j]))
				x[j] = min(ss.loc[j], w1[i][j])
			H[i][s_index] = max(x)+b1[i][0]

	return H
#========================================================================
def fuzzy_erosion(w1, b1, samples):
	
	tempH = np.dot(w1, samples) + b1
	H = np.ones((np.size(w1,0), np.size(samples,1)))
	
	for s_index in range(np.size(samples,1)):
		ss = samples.loc[:,s_index]
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
	
#========================================================================
def setOpts(argv):                         
	parser = argparse.ArgumentParser()
	parser.add_argument('-ds', '--Dataset', dest='Dataset', action='store', required=False,
						help="Filename of whole dataset")
	parser.add_argument('-tr', '--TrainingData_File',dest='TrainingData_File',action='store',required=False,
		help="Filename of training data set")
	parser.add_argument('-ts', '--TestingData_File',dest='TestingData_File',action='store',required=False,
		help="Filename of testing data set")
	parser.add_argument('-ty', '--Elm_Type',dest='Elm_Type',action='store',required=True,
		help="0 for regression; 1 for (both binary and multi-classes) classification")
	parser.add_argument('-nh', '--nHiddenNeurons',dest='nHiddenNeurons',action='store',required=True,
		help="Number of hidden neurons assigned to the OSELM")
	parser.add_argument('-af', '--ActivationFunction',dest='ActivationFunction',action='store', 
		help="Type of activation function:")
	parser.add_argument('-sd', '--seed',dest='nSeed',action='store', 
		help="random number generator seed:")
	parser.add_argument('-v', dest='verbose', action='store_true',default=False,
		help="Verbose output")
	parser.add_argument('-el', dest='error_limit', action='store', default=5e-5,
						help="Error Limit")
	parser.add_argument('-db', dest='func', action='store', default='uci',
						help="DatabaseSource")
	arg = parser.parse_args()
	return(arg.__dict__['TrainingData_File'], arg.__dict__['TestingData_File'], arg.__dict__['Elm_Type'], arg.__dict__['nHiddenNeurons'],
		arg.__dict__['ActivationFunction'], arg.__dict__['nSeed'], arg.__dict__['verbose'], arg.__dict__['error_limit'], arg.__dict__['Dataset'],
	arg.__dict__['func'])
#========================================================================
if __name__ == "__main__":

	opts = setOpts(sys.argv[1:])
	ff = authoral_melm(float(opts[7]))
	#print(opts[7])
	i = 2
	iteration_limit = 15
	#train_data = processCSV(opts[0])
	#test_data = processCSV(opts[1])
	if opts[9] == 'uci': func = 'process_uci_dataset'
	elif opts[9] == 'wine' : func = 'processCSV'
	else: func = 'processCSV'
	print(opts[9])
	train_data, test_data = MakeTrainTest(globals()[func](opts[8]), 0.9)
	print('iteration 1')
	new_train, new_test = ff.main(train_data, test_data, opts[2], opts[3], opts[4], opts[5], opts[6],
								  False, False)
	new_train, new_test = new_train.reset_index(drop=True), new_test.reset_index(drop=True)
	new_train.columns, new_test.columns = range(new_train.columns.size), range(new_test.columns.size)

	while True:
		print(f'iteration {i}:')
		new_train, new_test = ff.main(new_train, new_test, opts[2], opts[3], opts[4], opts[5], opts[6], False, False)
		new_train, new_test = new_train.reset_index(drop=True), new_test.reset_index(drop=True)
		new_train.columns, new_test.columns = range(new_train.columns.size), range(new_test.columns.size)
		i += 1
		if i == iteration_limit: print(f"""--------------\n
                                 Iteration has reached its limit ({iteration_limit})
                                 \n--------------"""); break
		if new_train.size == 0 and new_test.size == 0 : print('Sample size has become too small'); break


	print('last iteration, with all hidden neurons combined')
	new_train, new_test = ff.main(train_data, test_data, opts[2], opts[3], opts[4], opts[5], opts[6],
									  True, False)

	#run source mELM to compare with authoral
	gg = melm()
	gg.main(train_data, test_data, opts[2], ff.getWeightsLength(), opts[4], opts[5],opts[6], False)

	x = ff.getInputOutput()
	y = gg.getInputOutput()
	graph([ [(x[0],'Desired Output'),(x[1],'Output')],
			[(y[0],'Desired Output'),(y[1],'Output')] ],
		  ['authoral mELM', 'state of the art'],'Treinamento')
	graph([[(x[2], 'Desired Output'), (x[3], 'Output')],
		   [(y[2], 'Desired Output'), (y[3], 'Output')]],
		  ['authoral mELM', 'state of the art'], 'Teste')
	print(f"Autoral:\n Treino:{ff.getLastAccuracy()[0]} ; Teste: {ff.getLastAccuracy()[1]}")
	print(f"Estado da arte:\n Treino: {gg.getLastAccuracy()[0]} ; Teste: {gg.getLastAccuracy()[1]}")

#========================================================================
