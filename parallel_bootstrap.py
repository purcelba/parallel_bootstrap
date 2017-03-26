import pandas as pd
import numpy as np
import json

"""
A set of functions for analyzing output from sparkBoot and bashBoot.
- txt2df:    convert the output of bashBoot and sparkBoot to a pandas dataframe for subsequent analysis.
- predict:   given a set of observed features, predict the target variable using each bootstrap fit
- rmse_dist: compute root mean squared error for each bootstrapped sample
- r2_dist: compute coefficient of determination (r-squared) for each bootstrapped sample
"""

def txt2df(output_file):
    """
    Read the txt file input from sparkBoot.py in JSON format.  Convert to pandas dataframe.
    
    Parameters
    ----------
    output_file, str
        - Name of the output file from sparkBoot.py or bashBoot.py.
    
    Returns
    -------
    df_coef
        - A dataframe where each column is a feature and rows are bootstrapped parameter values for that feature.  
          The intercept is included as a column.
    df_info
        - A dataframe with columns containing additional information about the bootstrap including r2, mse_min,
          cv, sklearn_version, random_state, and training data.
    """
    
    #open the results
    boot_out_txt = open(output_file,'r')
    #initialize coefficient dataframe as null
    df_coef = pd.DataFrame(columns = [])
    df_info = pd.DataFrame(columns = [])
    #loop through lines of input, format it, and save it in the dataframes
    for line in boot_out_txt:
        #find only the dict in the output.
        output_dict = json.loads(line)
        #split the coefficient list into separate keys
        coef_dict = {}
        for idx,val in enumerate(output_dict['coef']):
            coef_dict[str(output_dict['column_names'][idx])] = [val]
        coef_dict['intercept'] = [output_dict['intercept']]
        #if a coef dataframe does not exist, create one with this dict
        if df_coef.empty:
            df_coef = pd.DataFrame(coef_dict)
        else:
            df_coef = pd.concat([df_coef,pd.DataFrame(coef_dict)])
        #save remaining variables in another dataframe (exclude intercept)
        var_list = ['r2','mse_min','cv','sklearn_version','random_state','training_data','alpha','training_labels','seed']
        df_dict = {}
        for v in var_list:
            df_dict[v] = [output_dict[v]]
        if df_info.empty:
            df_info = pd.DataFrame(df_dict)
        else:
            df_info = pd.concat([df_info,pd.DataFrame(df_dict)])
    return df_coef, df_info


def predict(df_coef,df_feats):
    """
    Given the dataframe of bootstrapped coefficients (b x m) and a dataframe of features (n x m), 
    compute the prediction for each observation and bootstrap (n x b).
    
    where b = number of bootstrap samples
          m = number of features
          n = number of observations
    """
    #sort columns
    df_feats = df_feats.sort_index('columns')
    df_coef = df_coef.sort_index('columns')
    #drop intercept from coefficients
    intercept = df_coef['intercept']
    df_coef = df_coef.drop('intercept', axis = 1)
    #predictions are just matrix multiplication plus intercept
    pred = df_feats.dot(df_coef.T) + intercept
    return pred
    

def rmse_dist(true,pred):
    """
    Given the true value of the target variable (n x 1) and a matrix of 
    predictions for each bootstrap (n x b). Compute the rmse for each 
    bootstrap.  Return a vector (1 x b) of the resulting rmse values.
    
    where,
        b = number of bootstrap samples
        n = number of observations
    
    Parameters
    ----------
    true, list; pandas series; or numpy array
        - target variable (n x 1)
    pred, dataframe of predictions (n x b)
        - Vector of predicted values.  Obtained from 'predict' function.
    
    """
    #b, num bootstraps and n, num oservations
    b = np.shape(pred)[1]
    n = np.shape(pred)[0]
    #tile the true values for efficient vector computation of rmse
    true = np.tile(np.array([true]).T,(1,b))
    #compute rmse
    rmse_dist = np.sqrt((1.0/n)*np.sum((true - np.array(pred))**2,axis=0))
    #return result
    return rmse_dist


def r2_dist(true,pred):
    """
    Given the true value of the target variable (n x 1) and a matrix of 
    predictions for each bootstrap (n x b). Compute the r2 for each 
    bootstrap.  Return a vector (1 x b) of the resulting rmse values.
    
    where,
        b = number of bootstrap samples
        n = number of observations
    
    Parameters
    ----------
    true, list; pandas series; or numpy array
        - target variable (n x 1)
    pred, dataframe of predictions (n x b)
        - Vector of predicted values.  Obtained from 'predict' function.
    """
    #b, num bootstraps and n, num oservations
    b = np.shape(pred)[1]
    n = np.shape(pred)[0]
    #tile the true values for efficient vector computation of rmse
    true = np.tile(np.array([true]).T,(1,b))
    #compute r2 from sum sq err and sum sq tot
    sse = np.sum((true - np.array(pred))**2,axis=0)
    means = np.tile(np.array([np.mean(true,axis=0)]),(n,1)) #tile the means
    sst = np.sum((true - means)**2,axis=0)
    r2 = 1-(sse/sst)
    #return
    return r2
  
