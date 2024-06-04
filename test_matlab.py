#scipy.test

from scipy.io import loadmat
import numpy as np

#Criar arquivos csv, separando o benigno e maligno.

# Carregar o arquivo .mat
data = loadmat('matlab/entrada_temp.mat')


benign = np.array([d for d in data['entrada_temp'] if d[-1] == 1])
malign = np.array([d for d in data['entrada_temp'] if d[-1] == 2])
breakpoint()
for d in benign:
    d[-1] = 0
for d in malign:
    d[-1] = 1
#only move last column to first and keep all the others the same
benign = np.concatenate((benign[:,[-1]], benign[:,:-1]), axis=1)
malign = np.concatenate((malign[:,[-1]], malign[:,:-1]), axis=1)

np.savetxt('matlab/benign_matlab.csv', benign, delimiter=';')
np.savetxt('matlab/malign_matlab.csv', malign, delimiter=';')
# Exibir as variáveis contidas no arquivo .mat
# 1 -> 
# 2 -> 