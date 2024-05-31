#scipy.test

from scipy.io import loadmat
import numpy as np

# Carregar o arquivo .mat
data = loadmat('matlab/entrada_temp.mat')

# Exibir as variáveis contidas no arquivo .mat
breakpoint()
print(np.size(data['entrada_temp'],0))
# 1 -> 
# 2 -> 