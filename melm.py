
from random import *
from time import process_time
from scipy import stats
from melm_lib import *
from collections import Counter
import copy
import math
import struct
import sys
import argparse
import numpy as np
import pandas as pd
import struct
from collections import Counter

np.set_printoptions(threshold=np.inf)
# ========================================================================
class melm():
    def __init__(self):
        self.weights = []
        self.biases = []
        self.inputOutput = []
        self.originalData_P = []
        self.originalData_T = []
        self.originalData_TVT = []
        self.originalData_TVP = []
        self.originalTrainData = []
        self.originalTestData = []
        self.accuracies = []
        self.saturated_neurons = []
        self.iteration = 0
        self.weight_labels = []
        
    def main(self,train_data, test_data,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,verbose, lastRun , plot = False):
        
        if ActivationFunction is None:
            ActivationFunction = 'linear'
        print('kernel ' + ActivationFunction)

        if nSeed == None:
            nSeed = 1
        seed(nSeed)
        np.random.seed(nSeed)

        Elm_Type = int(Elm_Type)
        NumberofHiddenNeurons = int(NumberofHiddenNeurons)
        if verbose: print ('Macro definition')
        #%%%%%%%%%%% Macro definition
        self.REGRESSION=0
        self.CLASSIFIER=1

        train_data_2 = train_data.copy()
        test_data_2 = test_data.copy()
        
        if verbose: print ('Load training dataset')
        #%%%%%%%%%%% Load training dataset
        T=np.transpose(train_data.loc[:,0]) # getting the training labels (1.0 or 0.0)
        P=np.transpose(train_data.loc[:,1::]) # getting the training features
        T = T.reset_index(drop=True)
        P = P.reset_index(drop=True)
                                 #%   Release raw training data array
        if verbose: print ('Load testing dataset')
        #%%%%%%%%%%% Load testing dataset
        TVT=np.transpose(test_data.loc[:,0]) # getting the testing labels (1.0 or 0.0)
        TVP=np.transpose(test_data.loc[:,1::]) # getting the testing features
        TVT = TVT.reset_index(drop=True)
        TVP = TVP.reset_index(drop=True)
        del(test_data)                                    #%   Release raw testing data array

        NumberofTrainingData=np.size(P,1)
        NumberofTestingData=np.size(TVP,1)
        NumberofInputNeurons=np.size(P,0)


        if Elm_Type!=self.REGRESSION:
            if verbose: print ('Preprocessing the data of classification')
            #%%%%%%%%%%%% Preprocessing the data of classification
            #%%%%%%%%%%%% Obtendo número de classes da base de dados
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
            #%%%%%%%%%%
            temp_T=np.zeros((NumberofOutputNeurons, NumberofTrainingData))
            
            for i in range(0, NumberofTrainingData): # Para cada amostra
                for j in range(0, number_class): # Para cada classe
                    if label[j] == T[i]: #Se a label estiver dentro das labels da base de dados
                        break
                temp_T[j][i]=1
            T=temp_T*2-1
            # T se transforma em uma matriz 2xN, onde N é o número de amostras
            # linha 0: se for 1, é da classe 1, se for -1, é classe 0
            # linha 1: se for 1, é da classe 0, se for -1, é classe 1
            
            if len(self.originalData_T) == 0:
                self.originalData_T = T
                self.originalData_P = P
            
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

        if verbose: print ('Random generate input weights inputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons')
        #%%%%%%%%%%% Random generate input weights inputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons

        if (ActivationFunction == 'erosion') or (ActivationFunction == 'dilation') or (ActivationFunction == 'fuzzy-erosion') or (ActivationFunction == 'fuzzy_erosion') or (ActivationFunction == 'fuzzy-dilation') or (ActivationFunction == 'fuzzy_dilation') or (ActivationFunction == 'bitwise-erosion') or (ActivationFunction == 'bitwise_erosion') or (ActivationFunction == 'bitwise-dilation') or (ActivationFunction == 'bitwise_dilation') :
            if lastRun == False:
                inputWeight = get_weights(self, train_data, label)
                BiasofHiddenNeurons = np.zeros((len(label),1))
                
            else:
                inputWeight = self.weights
                BiasofHiddenNeurons = self.biases

        else:
            inputWeight=np.random.rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;

        if lastRun == False:
            if np.array_equal(self.weights, []): 
                self.weights = inputWeight
                self.biases = BiasofHiddenNeurons
            else: 
                self.weights = np.row_stack((self.weights, inputWeight))
                self.biases = np.row_stack((self.biases, BiasofHiddenNeurons))


        if verbose: print ('Calculate hidden neuron output matrix H')   
             #%%%%%%%%%%% Calculate hidden neuron output matrix H
        
        #getting original P and T
        if self.iteration == 0:
            self.originalData_P = P
            self.originalData_T = T 
            self.originalData_TVT = TVT
            self.originalData_TVP = TVP
            self.originalTrainData = train_data_2
            self.originalTestData = test_data_2
            self.saturated_neurons = np.zeros((NumberofOutputNeurons,NumberofInputNeurons))

        H = switchActivationFunction(ActivationFunction,self.weights,self.biases,self.originalData_P)
        if verbose: print ('Calculate output weights OutputWeight (beta_i)')
        #%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
        pinvs = np.linalg.pinv(np.transpose(H))
        greatest_interval = [abs(max(v)  - min(v)) for v in pinvs]
        
        OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(self.originalData_T))

        end_time_train =  process_time()
        TrainingTime=end_time_train-start_time_train       # %   Calculate CPU time (seconds) spent for training ELM

        if verbose: print ('Calculate the training accuracy')
        #%%%%%%%%%%% Calculate the training accuracy
        Y = np.transpose(np.dot(np.transpose(H), OutputWeight))                     #%   Y: the actual output of the training data
        TrainingAccuracy = 0
        if Elm_Type == self.REGRESSION:
            if verbose: print ('Calculate training accuracy (RMSE) for regression case')
            #   Calculate training accuracy (RMSE) for regression case
            TrainingAccuracy = np.square(np.subtract(self.originalData_T, Y)).mean()
            TrainingAccuracy = round(TrainingAccuracy, 6)
            print('Training Accuracy: ' + str(TrainingAccuracy)+' ( ' + str(np.size(Y,0)) + ' samples) (regression)')
        #del(H)

        if verbose: print ('Calculate the output of testing input')
        start_time_test = process_time()
        #%%%%%%%%%%% Calculate the output of testing input

        tempH_test = switchActivationFunction(ActivationFunction,self.weights,self.biases,self.originalData_TVP)
        del(TVP)
        TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))                     #%   Y: the actual output of the training data
        
        if plot: graph([ [(T,'Saída Desejada'),(Y,'Saída Obtida')],
                    [(TVT,'Saída Desejada'),(TY,'Saída Obtida')] ],
                [ 'Treinamento', 'Teste'] ,
                'authoral mELM')
        self.inputOutput = [np.argmax(self.originalData_T,axis=0),
                    np.argmax(Y,axis=0),
                    np.argmax(self.originalData_TVT,axis=0),
                    np.argmax(TY,axis=0)]
        
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
            correctIndexes= np.array([])
            incorrectIndexes = np.array([])
            label_index_expected = np.argmax(self.originalData_T, axis=0)   # Maxima along the second axis
            label_index_actual = np.argmax(Y, axis=0)   # Maxima along the second axis

            for i in range(0, np.size(label_index_expected,0)):
                    if label_index_actual[i]!=label_index_expected[i]:
                            MissClassificationRate_Training=MissClassificationRate_Training+1
                            incorrectIndexes = np.append(incorrectIndexes, i)   
                    else:
                        correctIndexes = np.append(correctIndexes, i)
            train_data_2 = self.originalTrainData.drop(correctIndexes)
            train_data_2 = self.originalTrainData.reset_index(drop=True)
            TrainingAccuracy = 1-MissClassificationRate_Training/np.size(label_index_expected,0)
            TrainingAccuracy = round(TrainingAccuracy, 6)
            print('Training Accuracy: ' + str(TrainingAccuracy*100)+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Training),'/',str(np.size(label_index_expected,0)),') (classification)')
            
            label_index_expected = np.argmax(self.originalData_TVT, axis=0)   # Maxima along the second axis
            label_index_actual = np.argmax(TY, axis=0)   # Maxima along the second axis
            CorrectIndexesTest= np.array([])
            for i in range(0, np.size(label_index_expected,0)):
                if label_index_actual[i]!=label_index_expected[i]:
                            MissClassificationRate_Testing=MissClassificationRate_Testing+1
                else:
                        CorrectIndexesTest = np.append(CorrectIndexesTest, i)
            TestingAccuracy=1-MissClassificationRate_Testing/np.size(label_index_expected,0)
            TestingAccuracy = round(TestingAccuracy, 6)
            test_data_2 = self.originalTestData.drop(CorrectIndexesTest)
            
            dSet2 = pd.concat([train_data_2, test_data_2], ignore_index=True)

            print('Testing Accuracy: ' + str(TestingAccuracy*100)+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Testing),'/',str(np.size(label_index_expected,0)),') (classification)')

            print('Training Time: ' + str(round(TrainingTime,6)) + ' seconds')
            print('Testing Time: ' + str(round(TestingTime,6)) + ' seconds')

            self.clear_log()
            self.save_matrix_snippet(inputWeight, 'inputWeight')
            self.save_matrix_snippet(H, 'H', 10,10)
            self.save_matrix_snippet(Y, 'Y', 10,10)
            self.accuracies.append([TrainingAccuracy*100, TestingAccuracy*100])
            if self.iteration == 10 or self.iteration == 20: breakpoint()
            self.iteration +=1

            return dSet2

    
    def is_saturated(self, column_index, classe):
        #method that checks if a neuron is saturated
        #column_index: index of the column to be checked
        if self.iteration == 0: return False
        if self.saturated_neurons[classe][column_index] == 1: return True
        return False 


    def save_matrix_snippet(self, data, dataname, number_of_lines=0, number_of_columns=0):
        #method that saves only 10 first lines of a matrix  
        data_to_print = []
        if number_of_lines == 0 and number_of_columns == 0: data_to_print = data
        else:
            for index, line in enumerate(data):
                data_to_print.append(line[:number_of_columns])
                if index == number_of_lines:
                    break
            data = np.array(data_to_print)
        
        file = open("log.txt", "a")
        file.write(dataname + '\n')
        file.write('------------------------------------\n')
        np.savetxt(file, data, fmt='%s')
        file.write('------------------------------------\n\n')
        file.close()

    def clear_log(self):
        file = open("log.txt", "w")
        file.write('')
        file.close()

    def get_accuracies(self):
        return self.accuracies
    
    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.biases
    
    def getInputOutput(self):
        return self.inputOutput

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

def main() -> None:
    Values = []
    Accuracies = []
    opts = setOpts(sys.argv[1:])
    ## Naming Parameters
    benign_path, malign_path = opts[0], opts[1]
    elm_type = opts[2] #0 for regression; 1 for classification
    nHiddenNeurons = opts[3] #not utilized for authoral mELM
    activationFunction = opts[4] #always dilation for authoral mELM
    seed = opts[5] #seed for random number generator
    verbose = opts[6] #verbose output
    
    ff = melm() #calling melm class
    if 'matlab' in benign_path:
        entrada_maligno = [
    [-4.2308, -0.7077, 2.0000],
    [-3.7179, -0.5795, 2.0000],
    [-3.4615, -0.5654, 2.0000],
    [-2.6923, -0.3731, 2.0000],
    [-2.4359, -0.2590, 2.0000],
    [-1.9231, -0.1808, 2.0000],
    [-0.6410, 0.1897, 2.0000],
    [-0.1282, 0.2679, 2.0000],
    [0.1282, 0.3821, 2.0000],
    [0.3846, 0.4462, 2.0000],
    [1.1538, 0.6385, 2.0000],
    [1.4103, 0.6526, 2.0000],
    [2.1795, 0.8949, 2.0000],
    [2.6923, 0.9731, 2.0000],
    [2.9487, 1.0872, 2.0000],
    [3.4615, 1.1654, 2.0000],
    [3.7179, 1.2295, 2.0000],
    [4.2308, 1.3577, 2.0000],
    [4.7436, 1.4859, 2.0000],
    [5.0000, 1.6000, 2.0000],
    [-0.6410, 0.1258, 2.0000],
    [-0.6410, -0.4492, 2.0000],
    [-0.1282, 0.1429, 2.0000],
    [-0.1282, -0.2688, 2.0000],
    [0.1282, 0.3046, 2.0000],
    [0.1282, -0.3313, 2.0000],
    [0.3846, 0.3671, 2.0000],
    [0.3846, -0.2262, 2.0000],
    [1.1538, 0.3885, 2.0000],
    [1.1538, -0.1068, 2.0000],
    [1.4103, 0.5226, 2.0000],
    [1.4103, -0.0653, 2.0000],
    [2.1795, 0.6565, 2.0000],
    [2.1795, 0.0821, 2.0000],
    [2.1795, -0.5191, 2.0000],
    [2.6923, 0.8361, 2.0000],
    [2.6923, 0.1414, 2.0000],
    [2.6923, -0.4009, 2.0000],
    [2.9487, 0.9121, 2.0000],
    [2.9487, 0.2071, 2.0000],
    [2.9487, -0.3080, 2.0000],
    [3.4615, 0.7439, 2.0000],
    [3.4615, 0.3336, 2.0000],
    [3.4615, -0.2236, 2.0000],
    [3.7179, 0.9973, 2.0000],
    [3.7179, 0.3186, 2.0000],
    [3.7179, -0.1810, 2.0000],
    [4.2308, 1.2098, 2.0000],
    [4.2308, 0.4533, 2.0000],
    [4.2308, -0.0996, 2.0000],
    [4.7436, 0.8352, 2.0000],
    [4.7436, 0.4906, 2.0000],
    [4.7436, -0.0291, 2.0000],
    [5.0000, 0.8321, 2.0000],
    [5.0000, 0.4731, 2.0000],
    [5.0000, 0.0344, 2.0000],
    [5.0000, -0.6090, 2.0000]
]
        entrada_benigno = [
    [-5.0000, -0.4500, 1.0000],
    [-4.4872, -0.3718, 1.0000],
    [-3.4615, -0.1154, 1.0000],
    [-2.6923, 0.1269, 1.0000],
    [-2.4359, 0.1410, 1.0000],
    [-1.9231, 0.3192, 1.0000],
    [-1.6667, 0.3833, 1.0000],
    [-1.4103, 0.4474, 1.0000],
    [-0.6410, 0.5897, 1.0000],
    [-0.1282, 0.7179, 1.0000],
    [0.1282, 0.7821, 1.0000],
    [0.3846, 0.8962, 1.0000],
    [0.6410, 0.9603, 1.0000],
    [1.1538, 1.0885, 1.0000],
    [1.4103, 1.1026, 1.0000],
    [2.1795, 1.2949, 1.0000],
    [2.6923, 1.4731, 1.0000],
    [2.9487, 1.5372, 1.0000],
    [4.4872, 1.9218, 1.0000],
    [4.7436, 1.9859, 1.0000],
    [-5.0000, 0.5224, 1.0000],
    [-5.0000, 1.6459, 1.0000],
    [-4.4872, 0.5410, 1.0000],
    [-4.4872, 1.7133, 1.0000],
    [-3.4615, 0.8727, 1.0000],
    [-3.4615, 1.9073, 1.0000],
    [-2.6923, 0.1600, 1.0000],
    [-2.6923, 1.1520, 1.0000],
    [-2.4359, 0.1794, 1.0000],
    [-2.4359, 1.4195, 1.0000],
    [-1.9231, 0.3765, 1.0000],
    [-1.9231, 1.4374, 1.0000],
    [-1.6667, 0.5440, 1.0000],
    [-1.6667, 1.7356, 1.0000],
    [-1.4103, 0.6539, 1.0000],
    [-1.4103, 1.7192, 1.0000],
    [-0.6410, 0.7125, 1.0000],
    [-0.6410, 1.6834, 1.0000],
    [-0.1282, 1.0353, 1.0000],
    [0.1282, 1.1112, 1.0000],
    [0.3846, 1.2486, 1.0000],
    [0.6410, 1.3524, 1.0000],
    [1.1538, 1.4001, 1.0000],
    [1.4103, 1.2429, 1.0000],
    [2.1795, 1.4656, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000],
    [2.9487, 1.8026, 1.0000]
]
        
        df1 = pd.DataFrame(entrada_benigno).astype('float64')
        df2 = pd.DataFrame(entrada_maligno).astype('float64')
        dSet = pd.concat([df1, df2], ignore_index=True)
        dSet = dSet.reset_index(drop=True)
        dSet = dSet[[2,0,1]].reset_index(drop=True)
        dSet.columns = range(dSet.shape[1])
        breakpoint()
        [TrainingData, TestingData] = MakeTrainTest(dSet, 0.9)
    elif 'matlab' in benign_path:
        [TrainingData, TestingData] = MakeTrainTest(ProcessCSV_matlab(benign_path,malign_path), 0.9)
    else:
        [TrainingData, TestingData] = MakeTrainTest(ProcessCSV(benign_path,malign_path) , 0.9)
    OriginalTrainingData, OriginalTestingData = TrainingData.copy(), TestingData.copy()
    
    i = 0
    print("Iteration: ", i)
    Values.append(ff.main(TrainingData, TestingData, elm_type, 2, activationFunction, seed, verbose, False))
    while True:
        i+=1
        print("Iteration: ", i)
        [TrainingData, TestingData] = MakeTrainTest(Values[i-1] , 0.9)
        Values.append(ff.main(TrainingData, TestingData, elm_type, 2, activationFunction, seed, verbose, False ))
        
        #Condição de parada
        if ff.get_accuracies()[-1][0] == 100.0 and ff.get_accuracies()[-1][1] == 100.0 or i == 30 :
            i+=1
            print(f'Iteration {i} with Original Training Data:')
            Values.append(ff.main(OriginalTrainingData, OriginalTestingData, elm_type, 2*(i+1), activationFunction,
                                    seed, verbose,  True))
            break
        
        


    x = ff.getInputOutput()
    graph_classification( [[(x[0],'Desired Output'),(x[1],'Output')] ],
		  ['authoral mELM'],'Teste')
 
    print('Acurácias:', Accuracies)
    print('inputWeight:')
    matprint(ff.get_weights())
    print('Biases:')
    print(ff.get_biases())
    
    
if __name__ == "__main__":
    main()
#========================================================================

