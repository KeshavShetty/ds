# Required imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def calculate_vif(input_data, exclude_columns=[]):
    vif_df = pd.DataFrame( columns=['Feature','Vif'])
    x_vars = input_data.drop(exclude_columns, axis=1)
    xvar_names = x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared
        vif=round(1/(1-rsq),2)
        vif_df.loc[i]=[xvar_names[i],vif]
    return vif_df.sort_values(by='Vif', axis=0, ascending=False, inplace=False)

def standardize(x):
    return ((x-np.mean(x))/np.std(x))


def fit(df, dependent_column,
        p_value_cutoff = 0.01,
        vif_cutoff = 5,
        acceptable_r2_change = 0.02,
        scale_numerical = False,
        include_target_column_from_scaling = True,
        dummies_creation_drop_column_preference='dropFirst', # Available options dropFirst, dropMax, dropMin
        train_split_size = 0.7,
        max_features_to_select = 0,
        random_state_to_use=100) :    # max_features_to_select=0 means Select all fields

    data_for_auto_lr = df
    
    
    response_dictionary = {} # Store all return content in the dictionary
    
    # If any null or nan found in datafrane, then throw a exception and return.
    if data_for_auto_lr.isna().sum().sum()>0 :
        raise Exception('Data is not clean, Null or nan values found in few columns. Check with df.isnull().sum()')
        
    ## Convert Categorical variables 
    df_categorical = data_for_auto_lr.select_dtypes(include=['object', 'category'])
    
    categorical_column_names = df_categorical.columns
    numerical_column_names =  [i for i in data_for_auto_lr.columns if not i in df_categorical.columns] 
    
    for aCatColumnName in categorical_column_names:
        print(aCatColumnName)
        dummy_df = pd.get_dummies(data_for_auto_lr[aCatColumnName])
    
        if dummies_creation_drop_column_preference=='dropFirst' :
            dummy_df = dummy_df.drop(dummy_df.columns[0], 1)
        elif dummies_creation_drop_column_preference=='dropMax' :
            column_with_max_records = data_for_auto_lr[aCatColumnName].value_counts().idxmax()
            dummy_df = dummy_df.drop(column_with_max_records, 1)
        elif dummies_creation_drop_column_preference=='dropMin' :
            column_with_min_records = data_for_auto_lr[aCatColumnName].value_counts().idxmin()
            dummy_df = dummy_df.drop(column_with_min_records, 1)
        else :
            raise Exception('Invalid value passed for dummies_creation_drop_column_preference. Valid options are: dropFirst, dropMax, dropMin')
        data_for_auto_lr = pd.concat([data_for_auto_lr, dummy_df], axis=1)
        data_for_auto_lr.drop([aCatColumnName], axis=1, inplace=True)
    
    ## Scale numerical Feature scaling
    scaler = StandardScaler()
    if scale_numerical:
        numerical_column_names.remove(dependent_column)
        data_for_auto_lr[numerical_column_names] = scaler.fit_transform(data_for_auto_lr[numerical_column_names])
        if include_target_column_from_scaling==True:
            data_for_auto_lr[[dependent_column]] = scaler.fit_transform(data_for_auto_lr[[dependent_column]])
    
    # Model building starts here
    X = data_for_auto_lr.drop(dependent_column,1)
    y = data_for_auto_lr[dependent_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split_size , random_state=random_state_to_use)
    
    # Set max_features_to_select if not passed
    if max_features_to_select<=0 : # So no parameter passed, use all available column
        max_features_to_select = len(X_train.columns)
    
    # First run RFE
    lm = LinearRegression()
    rfe = RFE(lm,max_features_to_select) 
    rffilter = rfe.fit(X_train, y_train)
    columns_to_use_further = X_train.columns[rfe.support_]
    columns_to_use_further = columns_to_use_further.tolist()
    
    comment = 'After RFE(' + str(max_features_to_select) + ")" + str(columns_to_use_further)
    model_iteration_info = pd.DataFrame( columns=['comment','r2_train', 'r2_adj_train', 'rmse_test', 'r2_test'])
    model_iteration=0
    prev_adj_r2 = 1
    column_to_remove = ""
    retain_columns = []
    
    exit_loop = False;
    while(not exit_loop):
        # Backward Elimination using Vif and p-value
        
        # Chec k p-value using stat
        X_train_sm = X_train[columns_to_use_further]
        X_train_sm = sm.add_constant(X_train_sm)
        
        lm_1 = sm.OLS(y_train, X_train_sm).fit()
        print('R2=' + str(lm_1.rsquared) + ' R2Adj=' + str(lm_1.rsquared_adj) )
        
        # Lets predict on test data
        x_test_sm = X_test[columns_to_use_further]
        x_test_sm  = sm.add_constant(x_test_sm)
        y_pred_intr = lm_1.predict(x_test_sm)
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_intr))
        
        r2_test = r2_score(y_test, y_pred_intr)
        
        if model_iteration==0:
            prev_adj_r2 = lm_1.rsquared_adj
            model_iteration_info.loc[model_iteration]=[comment, lm_1.rsquared, lm_1.rsquared_adj, rmse_test, r2_test]
        else:
            change_in_adjr2 = prev_adj_r2 - lm_1.rsquared_adj
            if change_in_adjr2 > acceptable_r2_change : # Put back the column
                columns_to_use_further = columns_to_use_further + [column_to_remove]
                retain_columns = retain_columns + [column_to_remove]
                model_iteration_info.loc[model_iteration]=['Degraded Adj R2, Putting back feature '+str(column_to_remove)+"(" + comment + ")", lm_1.rsquared, lm_1.rsquared_adj, rmse_test, r2_test ]
                # rebuild the model by putting back the last removed column
                X_train_sm = X_train[columns_to_use_further]
                X_train_sm = sm.add_constant(X_train_sm)            
                lm_1 = sm.OLS(y_train, X_train_sm).fit()
            else :
                prev_adj_r2 = lm_1.rsquared_adj
                retain_columns = [] # Reset columns to retain in this iteration
                model_iteration_info.loc[model_iteration]=[comment, lm_1.rsquared, lm_1.rsquared_adj, rmse_test, r2_test]
                
        
        p_values_df = pd.DataFrame({'Feature':lm_1.pvalues.index, 'p-value':lm_1.pvalues.values}).sort_values(by='p-value', axis=0, ascending=False, inplace=False)
        p_values_df = p_values_df[(~p_values_df['Feature'].isin(retain_columns))]
        p_values_df_filtered = p_values_df[(p_values_df['p-value']>p_value_cutoff) & (p_values_df['Feature']!='const')]
        
        vif_df = calculate_vif(X_train[columns_to_use_further])
        vif_df = vif_df[(~vif_df['Feature'].isin(retain_columns))]
        vif_df_filtered = vif_df[(vif_df['Vif']>vif_cutoff)] # Already in asceinding order
        
        column_to_remove = ""
        for index, row in vif_df_filtered.iterrows():
            vif_with_high_value = row[0]        
            p_value_found = p_values_df_filtered[p_values_df_filtered['Feature']==vif_with_high_value]
            if len(p_value_found)>0 : # Common name found in feature in p-value df and vif df
                column_to_remove = vif_with_high_value
                break # No need to check further
        
        if column_to_remove=="" and len(p_values_df_filtered)>0: # There is no column column left in vif and df
            column_to_remove = p_values_df_filtered.iloc[0]['Feature']
            
        if column_to_remove=="" and len(vif_df_filtered)>0: # Still no column selected indicates all column are significant with low p-value
            column_to_remove = vif_df_filtered.iloc[0]['Feature']    
            print(vif_with_high_value)
        
        if column_to_remove!="":
            columns_to_use_further.remove(column_to_remove)
            comment = 'Removing ' + column_to_remove + ' with Vif=' + str(vif_df[(vif_df['Feature']==column_to_remove)].iloc[0]['Vif']) + ' p-value=' + str(p_values_df[(p_values_df['Feature']==column_to_remove)].iloc[0]['p-value'])
           
        if column_to_remove == "" or len(columns_to_use_further)==0:
            exit_loop = True
            
        model_iteration = model_iteration + 1
    
    # Final model and R2 on test data
    X_train_sm = X_train[columns_to_use_further]
    X_train_sm = sm.add_constant(X_train_sm)
    
    lm_1 = sm.OLS(y_train, X_train_sm).fit()
    
    # Lets predict on test data
    x_test_sm = X_test[columns_to_use_further]
    x_test_sm  = sm.add_constant(x_test_sm)
    y_pred_final = lm_1.predict(x_test_sm)
    
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_final))
    r2_test = r2_score(y_test, y_pred_final)
    
    model_iteration_info.loc[model_iteration]=['Final Model with '+str(columns_to_use_further), lm_1.rsquared, lm_1.rsquared_adj, rmse_test, r2_test]
    
    response_dictionary['model_iteration_info'] = model_iteration_info
    response_dictionary['r2_test'] = r2_test
    response_dictionary['p-values'] =  pd.DataFrame({'Feature':lm_1.pvalues.index, 'p-value':lm_1.pvalues.values}).sort_values(by='p-value', axis=0, ascending=False, inplace=False)

    response_dictionary['vif-values'] = calculate_vif(X_train[columns_to_use_further])
    
    response_dictionary['rmse_test'] = rmse_test
    
    return response_dictionary