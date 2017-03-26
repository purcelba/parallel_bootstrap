from pyspark import SparkContext
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
import sys
import time

#run in local mode?  Set to True for debugging on local machine.
local_mode=False

#initialize spark context
if not local_mode:
    #increase the memory limit
    SparkContext.setSystemProperty('spark.executor.memory','12g')
    SparkContext.setSystemProperty('spark.driver.memory','12g')
    SparkContext.setSystemProperty('spark.executor.memoryOverhead','12g')
    SparkContext.setSystemProperty('spark.driver.memoryOverhead','12g')
    #SparkContext.setSystemProperty('spark.executor.instances','1000') #note: cap seems to be 298 on hpc dumbo
    #create SparkContext
    sc = SparkContext(appName="sparkBoot")

def main(feats_name, targets_name, model_name, n_boot, seed_start, output_filename, train_test_flag):
    """
    Broadcasts the feats and target variables across executors.  
    Creates an RDD with n_boot partitions.  
    For each partition, a random sample is taken, with replacement, from the feats set and 
    a model is fitted.
    The resulting coefficients and some other fit information is saved as text in JSON format
    and outputted to a txt file.
    
    Parameters
    ----------
    feats_name, str
      - Name of the feats csv file (observations x features)
    targets_name, str
      - Name of the target variable csv file (column vector of observations)
    model_name, str
      - Can be one of the following:
        'linear' for non-regularized linear regression
        'lasso' for L1-regularized linear regression
        'ridge' for L2-regularized linear regression
    n_boot, int
      - Number of bootstrap samples to take
    seed_start, int
      - Where to begin prng seed.
    output_filename
      - Name of the output txt file to save
    
    """

    #load feats and targets
    input_dict = {}
    input_dict['feats'] = 'data/%s' % (feats_name)
    input_dict['targets'] = 'data/%s' % (targets_name)
    #load the feats and targets
    df = pd.read_csv("%s" % (input_dict['feats']))
    targets = pd.read_csv("%s" % (input_dict['targets']))
    #drop columns not used for prediction
    drop_cols = ["Unnamed: 0","index"]
    for dc in drop_cols:
        if dc in targets.columns:
            targets = targets.drop(dc,axis=1)
        if dc in df.columns:
            df = df.drop(dc,axis=1)
    #reduce to training or test set only if requested
    if (train_test_flag == 'train') and ('test_set' in df.columns):
        targets = targets[df['test_set'] == 0]
        df = df[df['test_set'] == 0]
    elif (train_test_flag == 'test') and ('test_set' in df.columns):
        targets = targets[df['test_set'] == 1]
        df = df[df['test_set'] == 1]
    df = df.drop('test_set', axis = 1)
            
    #broadcast the feats and targets
    df_b = sc.broadcast(df)
    targets_b = sc.broadcast(targets)

    #Set up the classifier.  3fold CV for selection of regularization term.
    if model_name == 'linear':
        model = LinearRegression(fit_intercept=True,
                                  normalize=False,
                                  copy_X=True,
                                  n_jobs=1)    
    elif model_name == 'lasso':
        model = LassoCV(alphas = [.05,.1,.2],
                         normalize = False,
                         fit_intercept = True,
                         verbose = False,
                         copy_X = False,
                         n_jobs = 3)
    elif model_name == 'ridge':
        model = RidgeCV(alphas = [.00001,.0001,.001,.01,.1,1,10,100,1000,10000],
                         normalize = False,
                         fit_intercept = True,
                         verbose = 1,
                         cv = 3)
    else:
        raise ValueError('model_name not recognized.')
        
    #Create an RDD that specifies prng seed to use
    samp_list = [(n,) for n in np.arange(seed_start, seed_start+n_boot)]
    samp_rdd = sc.parallelize(samp_list,n_boot)     #create RDD with one partition for each row (second arg is number of partitions)
    #Create a function that takes a tuple as input and returns 
    def func(tup):
        """
        Takes as input a tuple containing an integer.  The integer specifies the random seed that will be used to 
        randomly sample, with replacement, observations from the feats set provided.  The model is fitted to the 
        sampled feats.  Resulting best fit parameters, along with some other summary statistics and information are
        provided as input in a JSON string that will be written to the output file when all jobs are completed.
        
        Parameters
        ----------
        tup, rdd
          - series of tuples with different integer values defining the RNG seed to be used to sample observations
        
        Returns
        ----------
        tup[0], int
          - the seed that was used
        json.dumps(results_dict), str
          - dict in json format with the following keys:
            - alpha, the regularization term providing the best fit according to 3 fold cross-validation
            - random_state, the initial state used for fitting
            - training_feats, the name of the training_feats csv file
            - training_targets, the name of the target variable csv file
            - cv, the type of cross-validation used
            - sklearn_version, which version of sklearn was used
            - mse_min, the mean squared error for the test set on each fold
            - r2, the r-squared value (% var explained)
            - coef, parameter vector
            - intercept, intercept parameter
            - column_names, feature name corresponding to each parameter in the parameter vector
        """
        #take a random sample with replacement
        np.random.seed(seed=tup[0])                             #set the seed
        n_obs = np.shape(df_b.value)[0]                         #number of observations determines sample size
        samp = list(np.random.randint(0,high=n_obs,size=n_obs)) #draw the random sample with replacement
        #fit the model
        tic = time.time()
        results = model.fit(df_b.value.iloc[samp,:],np.ravel(targets_b.value.iloc[samp]))
        toc = tic - time.time()
        #save the results in a dict
        results_dict = {}
        results_dict['alpha'] = results.alpha_
        results_dict['random_state'] = results.random_state
        results_dict['training_feats'] = input_dict['feats']
        results_dict['training_targets'] = input_dict['targets']
        results_dict['cv'] = results.cv
        results_dict['sklearn_version'] = sklearn.__version__
        results_dict['mse_min'] = results.mse_path_.min()
        results_dict['r2'] = results.score(df_b.value.iloc[samp,:],np.ravel(targets_b.value.iloc[samp]))
        results_dict['coef'] = list(results.coef_)
        results_dict['intercept'] = results.intercept_
        results_dict['column_names'] = [i for i in df_b.value.columns]
        results_dict['fit_time'] = toc
        #convert results dict to json and save in tuple
        return(json.dumps(results_dict))

    #fit model in parallel
    results = samp_rdd.map(lambda p: func(p))
    #save to text file
    results.saveAsTextFile(output_filename)
    #stop the SparkContext.
    if not local_mode:
        sc.stop()

if __name__ == "__main__":

    #check for command line inputs. set defaults
    if len(sys.argv) > 1:
        feats_name = sys.argv[1]
    else:
        feats_name = 'reduced_feats.csv'      #name of feature matrix (csv file)
    if len(sys.argv) > 2:
        targets_name = sys.argv[2]
    else:
        targets_name = 'reduced_target.csv'   #name of target variable  
    if len(sys.argv) > 3:
        model_name = sys.argv[3]
    else:
        model_name = 'lasso'              
    if len(sys.argv) > 4:
        n_boot = int(sys.argv[4])
    else:
        n_boot = 2
    if len(sys.argv) > 5:
        seed_start = int(sys.argv[5])
    else:
        seed_start = 0   
    if len(sys.argv) > 6:
        output_filename = sys.argv[6]
    else:  
        output_filename = 'sparkBoot_output'
    if len(sys.argv) > 7:
        train_test_flag = sys.argv[7]
    else:
        train_test_flag = 'train'             #which data to fit? 'train' for training set, 'test' for test set, 'all' for all data   
    #execute main function      
    main(feats_name, targets_name, model_name, n_boot, seed_start, output_filename, train_test_flag)
