import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
import sys
import time
import os

def main(feats_name, targets_name, model_name, n_boot, seed_start, output_filename, train_test_flag):
    """
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
    output_filename, str
      - Name of the output txt file to save
    train_test_flag, str
      - 'train' for training feats only, 'test' for test feats only, 'all' for all feats
    
    """

    #load data
    print "Loading %s and %s..." % (feats_name,targets_name)
    input_dict = {}
    input_dict['feats'] = 'data/%s' % (feats_name)
    input_dict['targets'] = 'data/%s' % (targets_name)
    print "Loaded.\n"
    #load the data and targets
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
    if 'test_set' not in df.columns:
        print "WARNING: 'test_set' column not found, using all data."  
    if (train_test_flag == 'train') and ('test_set' in df.columns):
        targets = targets[df['test_set'] == 0]
        df = df[df['test_set'] == 0]
    elif (train_test_flag == 'test') and ('test_set' in df.columns):
        targets = targets[df['test_set'] == 1]
        df = df[df['test_set'] == 1]
    df = df.drop('test_set', axis = 1)
            
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
                         verbose = True,
                         copy_X = False)
    elif model_name == 'ridge':
        model = RidgeCV(alphas = [.00001,.0001,.001,.01,.1,1,10,100,1000,10000],
                         normalize = False,
                         fit_intercept = True,
                         verbose = 1,
                         cv = 3)
    else:
        raise ValueError('model_name not recognized.')
    #Get list of prng seeds to run
    for n in np.arange(seed_start, seed_start+n_boot):
        #take a random sample with replacement
        print "Seed = %d" % (n)
        np.random.seed(seed=n)                                  #set the seed
        n_obs = np.shape(df)[0]                                 #number of observations determines sample size
        samp = list(np.random.randint(0,high=n_obs,size=n_obs)) #draw the random sample with replacement
        #fit the model
        print "\tFitting model..."
        tic = time.time()
        results = model.fit(df.iloc[samp,:],np.ravel(targets.iloc[samp]))
        toc = time.time() - tic
        print "\tFinished. Time = %2.2f seconds." % (toc)
        #save the results in a dict
        results_dict = {}
        results_dict['seed'] = n
        results_dict['alpha'] = results.alpha_
        results_dict['random_state'] = results.random_state
        results_dict['training_feats'] = input_dict['feats']
        results_dict['training_targets'] = input_dict['targets']
        results_dict['cv'] = results.cv
        results_dict['sklearn_version'] = sklearn.__version__
        results_dict['mse_min'] = results.mse_path_.min()
        results_dict['r2'] = results.score(df.iloc[samp,:],np.ravel(targets.iloc[samp]))
        results_dict['coef'] = list(results.coef_)
        results_dict['intercept'] = results.intercept_
        results_dict['column_names'] = [i for i in df.columns]
        results_dict['fit_time'] = toc
        #write to output txt file
        print "\tWriting to %s..." % (output_filename)
        f = open(output_filename,'a')
        f.write(json.dumps(results_dict) + "\n")
        f.close()
        print "\tDONE."

if __name__ == "__main__":

    #turn local_mode on for debugging, off for use on cluster
    local_mode = True

    #if in local_mode, check for command line arguments
    if local_mode:
        if len(sys.argv) > 1:
            feats_name = sys.argv[1]
        else:
            feats_name = 'reduced_feats.csv'      #name of feature matrix (csv file)
        if len(sys.argv) > 2:
            targets_name = sys.argv[2]
        else:
            targets_name = 'reduced_target.csv'  #name of target variable  
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
            output_filename = 'bashBoot_output_reduced.txt'
        if len(sys.argv) > 7:
            train_test_flag = sys.argv[7]
        else:          
            train_test_flag = 'train'
                        
    #if not in local model, read global variables
    if not local_mode:
        feats_name = os.environ['FEATS_NAME']
        targets_name = os.environ['TARGETS_NAME']
        model_name = os.environ['MODEL_NAME']
        n_boot = int(os.environ['N_BOOT'])
        seed_start = int(os.environ['SEED_START'])
        output_filename = os.environ['OUTPUT_FILENAME']
        train_test_flag = os.environ['TRAIN_TEST_FLAG']
    else:
        print "LOCAL MODE ACTIVE"
        
    #print the inputs
    print "Input variables:"
    print "\t feats_name = %s" % (feats_name)
    print "\t targets_name = %s" % (targets_name)
    print "\t model_name = %s" % (model_name)
    print "\t n_boot = %s" % (n_boot)
    print "\t seed_start = %s" % (seed_start)
    print "\t output_filename = %s" % (output_filename)
    print "\t train_test_flag = %s" % (train_test_flag)
    #execute main function
    main(feats_name, targets_name, model_name, n_boot, seed_start, output_filename, train_test_flag)
