import pandas as pd
import numpy as np
from scipy import stats
from melm_R3 import ProcessCSV

np.set_printoptions(threshold=np.inf)

Malign_address = "malignos_v2.csv"
Benign_address = "benignos_v2.csv"
#df = pd.read_csv(Malign_address, sep=';', decimal=".", header=None, low_memory=False)

df = ProcessCSV(Malign_address, Benign_address)
corr_df = pd.DataFrame(columns=['r', 'p-value'])

for col in df:                                    # Use this to loop through the DataFrame
  if pd.api.types.is_numeric_dtype(df[col]):      # Only calculate r, p-value for the numeric columns
    r, p = stats.pearsonr(df[0], df[col])    # .pearsonr() returns two values in a list, store them individually using this format
                                              # r and p-values are calculated in comparison with what it is(benign or malign)
    corr_df.loc[col] = [round(r, 3), round(p, 3)] # Add the r & p for this col into the corr_df
corr_df = corr_df.sort_values(by=['p-value'], ascending=True)    # Sort and display the corr_df

np.savetxt('r and p-values.txt',corr_df)
print(corr_df)

condicao = corr_df['p-value'] < 0.2
indices = corr_df[condicao]
print(list(indices.index))
print(df[list(indices.index)])