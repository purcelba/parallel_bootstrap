#!/bin/bash

#Run multiple jobs bashBoot.py jobs in parallel.  This code will repeatedly call pbs_bashBoot.sh 
#with a sequence of different pseudo random number generator (PRNG) seeds used for sampling.
#Repeatedly calls bashBoot.pbs script to submit jobs.  Edit that pbs script if it is necessary
#to request different amounts of resources (e.g., longer/shorter walltime, more/less memory).
#See https://wikis.nyu.edu/display/NYUHPC/Running+jobs+on+the+NYU+HPC+clusters for more details
#about PBS scripts.
#
#Run from the command line as follows:
#   $./exe_bashBoot.sh data_name labels_name clf_name output_filename seed_start seed_end train_test_flag
# where
#   feats_name, str, name of the feature matrix csv file (observations x features).  Should be saved in /data subdirectory.
#   targets_name, str, name of the target variable column vector csv file (observations x 1).  Should be saved in /data subdirectory.
#   model_name, str, name of the classifier to be used.  Can be 'linear' for unregularized linear regression, 'lasso' for L1-regularized regression, or 'ridge' for L2-regularized regression.
#   output_filename, str, name of the output txt file in which to write the results.  If file exists already, then results will be appended on new lines.
#   seed_start, int, PRNG seed to start the sequence.
#   seed_end, int, PRNG seed to end the sequence.  Must be = or > seed_start.'
#   train_test_flag, str, if 'train' then using only training data, if 'test' use only test data
#

#name command line variables for clarity.
feats_name=$1            #Example: "inter_feats.csv"
targets_name=$2          #Example: "inter_target.csv"
model_name=$3             #Example: "lasso"
output_filename=$4      #Example: "bashBoot_output_df0_train.txt"
seed_start=$5           #Example: 1
seed_end=$6             #Example: 100
train_test_flag=$7    #'train'

#submit jobs in a loop
while [ $seed_start -le $seed_end ]
do
    qsub -v FEATS_NAME=$feats_name,TARGETS_NAME=$targets_name,MODEL_NAME=$model_name,OUTPUT_FILENAME=$output_filename,TRAIN_TEST_FLAG=$train_test_flag,SEED_START=$seed_start,N_BOOT=1 bashBoot.pbs
    seed_start=$(( $seed_start + 1 ))
done



