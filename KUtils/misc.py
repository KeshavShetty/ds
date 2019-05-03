# KUtils package

import numpy as np
import pandas as pd
import statsmodels.api as sm

def sayHello():
    ''' Dummy Function to print Hello '''
    print('Hello')

binaryPositiveLabel = 'Yes'
binaryNegativeLabel = 'No'

def setBinaryMapLabels(newBinaryPositiveLabel, newBinaryNegativeLabel): # Yes/No, Upper/Lower, True/False etc
    ''' Sets new value to be used in map_binary method
        call using setBinaryMapLabels('Yes', 'No') or setBinaryMapLabels(newBinaryPositiveLabel='Yes', newBinaryNegativeLabel='No') '''
    global binaryPositiveLabel
    global binaryNegativeLabel
    binaryPositiveLabel = newBinaryPositiveLabel
    binaryNegativeLabel = newBinaryNegativeLabel
    
def map_binary(x):
    '''
    
    '''
    global binaryPositiveLabel
    global binaryNegativeLabel
    return x.map({binaryPositiveLabel:1, binaryNegativeLabel:0})

def normalize(x):
    return ((x-np.mean(x))/(max(x)-min(x)))

def calculate_vif(input_data, exclude_columns):
    vif_df = pd.DataFrame( columns=['Var','Vif'])
    x_vars = input_data.drop(exclude_columns, axis=1)
    xvar_names = x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared
        vif=round(1/(1-rsq),2)
        vif_df.loc[i]=[xvar_names[i],vif]
    return vif_df.sort_values(by='Vif', axis=0, ascending=False, inplace=False)

    