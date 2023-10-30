
from random import *
from time import process_time
from scipy import stats

from test import remove_uniform_columns, get_duplicate_rows

import math
import struct
import sys,string
import argparse
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
# ========================================================================
class melm():
    def main(self,train_data, test_data,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,verbose, NeuronStack, BiasArray, lastRun):

        if ActivationFunction is None:
            ActivationFunction = 'linear'
        print('kernel ' + ActivationFunction)

        if nSeed is None:
            nSeed = 1
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

        train_data_2 = train_data.copy()
        test_data_2 = test_data.copy()

        T=np.transpose(train_data.loc[:,0])
        P=np.transpose(train_data.loc[:,1::])
        classValues = train_data.loc[train_data[0] == 1.000000]
        counterClassValues = train_data.loc[train_data[0] == 0.000000]
        T = T.reset_index(drop=True)
        P = P.reset_index(drop=True)
        classValues = (classValues.reset_index(drop=True))
        counterClassValues = train_data.loc[train_data[0] == 0.000000]
        counterClassValues = (counterClassValues.reset_index(drop=True))


        if verbose: print ('Load testing dataset')
        #%%%%%%%%%%% Load testing dataset

        #test_data=pd.read_csv(TestingData_File, sep=' ', decimal=".", header=None)
        #for ii in reversed(range(np.size(test_data,1))):
        #    if np.isnan(test_data.loc[:,ii]).all():
        #        test_data.drop(test_data.columns[ii], axis=1, inplace=True)

        TVT=np.transpose(test_data.loc[:,0])
        TVP=np.transpose(test_data.loc[:,1::])
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

        if (ActivationFunction == 'erosion') or (ActivationFunction == 'dilation') or (ActivationFunction == 'fuzzy-erosion') or (ActivationFunction == 'fuzzy_erosion') or (ActivationFunction == 'fuzzy-dilation') or (ActivationFunction == 'fuzzy_dilation') or (ActivationFunction == 'bitwise-erosion') or (ActivationFunction == 'bitwise_erosion') or (ActivationFunction == 'bitwise-dilation') or (ActivationFunction == 'bitwise_dilation') :
            #InputWeight = np.random.uniform(np.amin(np.amin(P)), np.amax(np.amax(P)), (NumberofHiddenNeurons,NumberofInputNeurons))
            if lastRun == False:
                #print('classValues',classValues, 'counterClassValues',counterClassValues ,sep='\n')
                classCalc = np.nanmean(classValues, axis = 0)                                               #Average
                counterClassCalc = np.nanmean(counterClassValues, axis = 0)                                 #Average
                #classCalc = np.nanmedian(classValues, axis = 0)                                            #Median
                #counterClassCalc = np.nanmedian(counterClassValues, axis = 0)                              #Median
                #classCalc = stats.mode(classValues, axis = 0 , keepdims= True , nan_policy= 'omit')[0]                      #Mode
                #counterClassCalc = stats.mode(counterClassValues, axis = 0 ,  keepdims= True ,  nan_policy= 'omit')[0]      #Mode
                #print('classCalc',classCalc, 'counterClassCalc',counterClassCalc ,sep='\n')
                classStd = np.nanstd(classCalc)                                                 #C Standard Deviation
                counterClassStd = np.nanstd(counterClassCalc)                                   #C.C Standard Deviation
                #print('classStd',classStd, 'counterClassStd',counterClassStd ,sep='\n')
                #print(classCalc, counterClassCalc)
                InputWeight = np.row_stack((classCalc, counterClassCalc))
                InputWeight = np.delete(InputWeight, 0, 1)
                InputWeight = np.row_stack((InputWeight, InputWeight, InputWeight))
                #print(InputWeight)

                BiasofHiddenNeurons = np.nan_to_num(
                    np.matrix([[0], [0], [classStd], [counterClassStd], [-classStd], [-counterClassStd]]), nan=0)
            else:
                InputWeight = NeuronStack
                BiasofHiddenNeurons = np.nan_to_num(BiasArray, nan=0)
                #print('NeuronStack: ', NeuronStack)
        else:
            InputWeight=np.random.rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;

        if np.array_equal(NeuronStack , []) :
            InputWeight = InputWeight
        else:
            #InputWeight = InputWeight + np.array(NeuronStack)
            InputWeight = InputWeight

        if verbose: print ('Calculate hidden neuron output matrix H')
        #%%%%%%%%%%% Calculate hidden neuron output matrix H
        print('ActivationFunction: ', ActivationFunction, 'InputWeight: ', InputWeight, 'BiasofHiddenNeurons: ', BiasofHiddenNeurons, 'P: ', P, sep='\n')
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
            TrainingAccuracy = np.square(np.subtract(T, Y)).mean()
            TrainingAccuracy = round(TrainingAccuracy, 6)
            print('Training Accuracy: ' + str(TrainingAccuracy)+' ( ' + str(np.size(Y,0)) + ' samples) (regression)')
        #del(H)

        if verbose: print ('Calculate the output of testing input')
        start_time_test = process_time()
        #%%%%%%%%%%% Calculate the output of testing input
        tempH_test = switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,TVP)
        del(TVP)

        TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))                     #%   Y: the actual output of the training data

        end_time_test = process_time()
        TestingTime=end_time_test-start_time_test           #%   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

        TestingAccuracy = 0
        if Elm_Type == self.REGRESSION:
            if verbose: print ('Calculate testing accuracy (RMSE) for regression case')
            #   Calculate testing accuracy (RMSE) for regression case
            TestingAccuracy = np.square(np.subtract(TVT, TY)).mean()
            TestingAccuracy = round(TestingAccuracy, 6)
            print('Testing Accuracy: ' + str(TestingAccuracy)+' ( ' + str(np.size(TY,0)) + ' samples) (regression)')

        if Elm_Type == self.CLASSIFIER:
            if verbose: print ('Calculate training & testing classification accuracy')
            #%%%%%%%%%% Calculate training & testing classification accuracy
            MissClassificationRate_Training=0
            MissClassificationRate_Testing=0
            CorrectIndexes= np.array([])
            label_index_expected = np.argmax(T, axis=0)   # Maxima along the second axis
            label_index_actual = np.argmax(Y, axis=0)   # Maxima along the second axis
            for i in range(0, np.size(label_index_expected,0)):
                    if label_index_actual[i]!=label_index_expected[i]:
                            MissClassificationRate_Training=MissClassificationRate_Training+1
                    else:
                        CorrectIndexes = np.append(CorrectIndexes, i)
            train_data_2 = train_data_2.drop(CorrectIndexes)
            train_data_2 = train_data_2.reset_index(drop=True)
            TrainingAccuracy = 1-MissClassificationRate_Training/np.size(label_index_expected,0)
            TrainingAccuracy = round(TrainingAccuracy, 6)
            print('Training Accuracy: ' + str(TrainingAccuracy*100)+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Training),'/',str(np.size(label_index_expected,0)),') (classification)')

            label_index_expected = np.argmax(TVT, axis=0)   # Maxima along the second axis
            label_index_actual = np.argmax(TY, axis=0)   # Maxima along the second axis
            CorrectIndexesTest= np.array([])
            for i in range(0, np.size(label_index_expected,0)):
                if label_index_actual[i]!=label_index_expected[i]:
                            MissClassificationRate_Testing=MissClassificationRate_Testing+1
                else:
                        CorrectIndexesTest = np.append(CorrectIndexesTest, i)
            TestingAccuracy=1-MissClassificationRate_Testing/np.size(label_index_expected,0)
            TestingAccuracy = round(TestingAccuracy, 6)
            test_data_2 = test_data_2.drop(CorrectIndexesTest)
            dSet2 = pd.concat([train_data_2, test_data_2], ignore_index=True)

            print('Testing Accuracy: ' + str(TestingAccuracy*100)+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Testing),'/',str(np.size(label_index_expected,0)),') (classification)')

            print('Training Time: ' + str(round(TrainingTime,6)) + ' seconds')
            print('Testing Time: ' + str(round(TestingTime,6)) + ' seconds')
            zipped = ['InputWeight:',InputWeight,'OutputWeight', OutputWeight,'Saída',Y]
            file = open("log.txt", "a")
            print(zipped)
            file.write('InputWeight\n')
            np.savetxt(file, InputWeight, fmt='%s')
            np.savetxt(file, InputWeight.shape, fmt='%s')
            file.write('H:\n')
            np.savetxt(file, H.shape, fmt='%s')
            np.savetxt(file, H, fmt='%s')
            file.write('OutputWeight\n')
            np.savetxt(file, OutputWeight, fmt='%s')
            np.savetxt(file, OutputWeight.shape, fmt='%s')
            file.write('Saída:\n')
            np.savetxt(file, Y, fmt='%s')
            np.savetxt(file, Y.shape, fmt='%s')
            return [dSet2, [TrainingAccuracy*100, TestingAccuracy*100], InputWeight, BiasofHiddenNeurons]
#========================================================================
def MakeTrainTest(dSet, percentTraining):
        #shuffle dataset
    dSet = dSet.sample(frac=1, random_state=349).reset_index(drop=True)
    splitPoint = int(len(dSet)*percentTraining)
    train_data, test_data = dSet[:splitPoint], dSet[splitPoint:]
    test_data = test_data.reset_index(drop=True)
    ##print(train_data, test_data)
    return [train_data, test_data]

def ProcessCSV(Benign_address , Malign_address , pvalue=0.2):
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

def euclidianDistance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

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
    x = np.zeros(np.size(w1,1))

    for s_index in range(np.size(samples,1)):
        ss = samples.loc[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
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
    parser.add_argument('-tr', '--TrainingData_File',dest='TrainingData_File',action='store',required=True,
        help="Filename of training data set")
    parser.add_argument('-ts', '--TestingData_File',dest='TestingData_File',action='store',required=True,
        help="Filename of testing data set")
    parser.add_argument('-ty', '--Elm_Type',dest='Elm_Type',action='store',required=True,
        help="0 for regression; 1 for (both binary and multi-classes) classification")
    parser.add_argument('-nh', '--nHiddenNeurons',dest='nHiddenNeurons',action='store',required=False,
        help="Number of hidden neurons assigned to the OSELM")
    parser.add_argument('-af', '--ActivationFunction',dest='ActivationFunction',action='store',
        help="Type of activation function:")
    parser.add_argument('-sd', '--seed',dest='nSeed',action='store',
        help="random number generator seed:")
    parser.add_argument('-v', dest='verbose', action='store_true',default=False,
        help="Verbose output")
    arg = parser.parse_args()
    return(arg.__dict__['TrainingData_File'], arg.__dict__['TestingData_File'], arg.__dict__['Elm_Type'], arg.__dict__['nHiddenNeurons'],
        arg.__dict__['ActivationFunction'], arg.__dict__['nSeed'], arg.__dict__['verbose'])
#========================================================================
if __name__ == "__main__":
    Values = []
    Dataset = []
    Accuracies = []
    Weights = np.array([])
    Biases = np.array([])
    opts = setOpts(sys.argv[1:])
    ff = melm()
    [TrainingData, TestingData] = MakeTrainTest(ProcessCSV(opts[0],opts[1]) , 0.7)
    OriginalTrainingData, OriginalTestingData = TrainingData.copy(), TestingData.copy()
    for i in range(999):
        print("Iteration: ", i)
        if i == 0:
            #Dataset.append(ff.main(opts[0], opts[1], opts[2], 2, opts[4], opts[5], opts[6],pd.DataFrame(), []))
            Dataset.append(ff.main(TrainingData.astype('float64'), TestingData.astype('float64'), opts[2], 6, opts[4], opts[5], opts[6], [], [] , False))
            Values.append(Dataset[i][0])
            Accuracies.append(Dataset[i][1])
            Weights = Dataset[i][2]
            Biases = Dataset[i][3]
        else:
            #Dataset.append(ff.main(opts[0], opts[1], opts[2], 2, opts[4], opts[5], opts[6],Values[i-1], []))
            [TrainingData, TestingData] = MakeTrainTest(Values[i-1] , 0.7)
            Dataset.append(ff.main(TrainingData, TestingData, opts[2], 6, opts[4], opts[5], opts[6], Weights[i-1], [] , False ))
            Values.append(Dataset[i][0])
            Accuracies.append(Dataset[i][1])
            #Weights.append(Dataset[i][2])
            Weights = np.row_stack((Weights,Dataset[i][2]))
            Biases = np.row_stack((Biases,Dataset[i][3]))
            if (Dataset[i][1])[0] == 100.0  and (Dataset[i][1])[1] == 100.0 or i == 5 :
                #print('Acurácia chegou em 100% no treinamento na iteração: ', i)
                #print(Weights)
                #print(Weights[~np.isnan(Weights).any(axis=1)])
                Dataset.append(ff.main(OriginalTrainingData.astype('float64'), OriginalTestingData.astype('float64'), opts[2], 6*(i+1), opts[4], opts[5], opts[6], np.nan_to_num(Weights, nan = 0), Biases, True))
                Values.append(Dataset[i][0])
                Accuracies.append(Dataset[i][1])

                dup = get_duplicate_rows(pd.DataFrame(Dataset[0][i]), pd.DataFrame(Dataset[0][0]))
                break

    print('Pesos: ', Weights)

    print('Acurácias:', Accuracies)

    print("Erros da ultima iteração que são iguais com o da primeira")
    print(dup)
#========================================================================

