import pandas as pd
import numpy as np

# noinspection PyTypeChecker
#np.set_printoptions(threshold=np.inf)
#Malign_address = "benignos_v2.csv"
#malign = pd.read_csv(Malign_address, sep=';', decimal=".", header=None, low_memory=False)
# = malign.drop([malign.columns[0], malign.columns[-1]], axis=1)
#malign = malign.drop([0])
#malign = malign.replace('0', -1)
#print(malign.replace(0,-1))
#malign = remove_uniform_columns(malign)
#print(malign)

def remove_uniform_columns(df):
        df = df.drop([df.columns[0], df.columns[-1]], axis=1)
        df = df.drop([0])
        # cols = df.select_dtypes([np.number]).columns
        # diff = df[cols].diff().abs().sum()
        # df = df.drop(diff[diff == 0].index, axis=1)
        df = df[[c for c
                         in list(df)
                         if len(df[c].unique()) > 1]]
        df.columns = range(df.columns.size)
        return df

def get_duplicate_rows(a,b):
    #a = pd.DataFrame([[1,2,3],[2,3,4],[4,5,6]])
    #b = pd.DataFrame([[1,2,3],[4,5,6],[6,7,8]])
    #print(a.iloc[0].equals(b.iloc[0]))
    df = pd.concat([a,b])
    return df[df.duplicated(keep = 'last')]

