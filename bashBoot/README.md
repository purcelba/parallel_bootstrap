# bashBoot

Parallelized bootstrapping using Torque cluster management software.

### Running bashBoot on the NYU HPC primary cluster (Mercer)

1. Log on to the Mercer cluster. See the [HPC wiki](https://wikis.nyu.edu/display/NYUHPC/Clusters+-+Mercer) for detailed instructions on connecting to the cluster.

2.  Move the bashBoot directory with code and data subdirectory to the computing server.  It should be placed in your scratch directory (e.g., /netID/scratch).  Make this your current directory.

3. Modify the PBS script, bashBoot.pbs.  Replace *netID* with your actual NYU NetID.  
```
#!/bin/bash
#PBS -l nodes=1:ppn=1,walltime=1:00:00
#PBS -l mem=10000MB
#PBS -N bashBoot
cd /scratch/netID/bashBoot/
module load python/intel/2.7.6
python </scratch/netID/bashBoot/bashBoot.py

```
Torque requires the user to specify the maximum run time for each job (walltime) and the memory allocation for each job.  The default walltime (1 hour) and memory allocation (10GB) should be plenty for most needs, but these can be editted as needed (see [HPC wiki](https://wikis.nyu.edu/display/NYUHPC/Running+jobs) for details).
4. Modify the permissions if needed.
```
chmod 755 bashBoot.sh
```
5. Run the jobs from the command line with the chosen parameters.
* *feats_name*, str, Name of the feature matrix csv file (observations x features)
* *targets_name*, str, Name of the target variable csv file (column vector of observations)
* *model_name*, str, Can be one of the following:
    * 'linear' for non-regularized linear regression
    * 'lasso' for L1-regularized linear regression
    * 'ridge' for L2-regularized linear regression
* *output_filename*, str, Name of the output txt file to save
* *seed_start*, int, Determines the start of the pseudo random number generator sequence.  Use different seeds for repeated bootstrap samples of the same model. Will submit jobs in a sequence with differnt seeds until reaching seed_end.
* *seed_end*, int, Last seed to be used.
* *train_test_flag*, str (optional), if the feature matrix includes a column to indicate training or test set observations, then indicate which set to use here.  Otherwise, all observations are sampled.
    * 'train' sample only training set data (test_set = 0)
    * 'test' sample only test set data (test_set = 1)
```
./bashBoot.sh 'feats_name' 'targets_name' 'model_name' 'output_filename' 'seed_start' 'seed_end' 'train_test_flag'
```
6. Monitor progress.  Use the qstat command with your NetID to monitor progress. Q indicates that the job is in the queue.  R indicates running. C indicates complete.
```
qstat -au NetID
```
7. After all jobs are completed, the output will be compiled in the *output_filename* text file. 
