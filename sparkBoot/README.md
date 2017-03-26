# sparkBoot

Parallelized bootstrapping using Apache Spark.

### Running sparkBoot on the NYU HPC Hadoop cluster (Dumbo)

1.  Log on to the Dumbo cluster.  See the [HPC wiki](https://wikis.nyu.edu/display/NYUHPC/Clusters+-+Dumbo) for detailed instructions on connecting to the cluster. 

2.  Move the sparkBoot directory with code and data to the computing server and make this the current directory.

3.  Load Python 2.7.11 that includes the necessary libraries (Numpy, SciPy, Pandas, scikit-learn).
```
module load python/gnu/2.7.11
```
4. Put the data directory on the Hadoop distributed file system (HDFS).
```
hadoop fs -put data
```
5. Check if output of the same name already exists.  If so, the job will not run.  
```
hadoop fs rm -r 'output_filename'
```
6. Submit the file with the chosen parameters.
    * *feats_name*, str, Name of the feature matrix csv file (observations x features)
    * *targets_name*, str, Name of the target variable csv file (column vector of observations)
    * *model_name*, str, Can be one of the following:
        * 'linear' for non-regularized linear regression
        * 'lasso' for L1-regularized linear regression
        * 'ridge' for L2-regularized linear regression
    * *n_boot*, int, Number of bootstrap samples to take
    * *seed_start*, int, Determines the start of the pseudo random number generator sequence.  Use different seeds for repeated bootstrap samples of the same model. 
    * *output_filename*, str, Name of the output txt file to save
    * *train_test_flag*, str (optional), if the feature matrix includes a column to indicate training or test set observations, then indicate which set to use here.  Otherwise, all observations are sampled.
        * 'train' sample only training set data (test_set = 0)
        * 'test' sample only test set data (test_set = 1)
```
spark-submit sparkBoot.py 'feats_name' 'targets_name' 'clf_name' ''n_boot' 'output_filename' 'train_test_flag'
```
7. Monitor progress
    - If the jobs have started correctly, then you should see a series of messages with some variation of "Added broadcast_#_piece# in memory...".
    - To montior progress, get the tracking URL from the printout paste it into a web browser.  For example, "tracking URL: http://babar.es.its.nyu.edu:8088/proxy/application_1484865967044_28476/".  Click on the links under "Description" until you come to a page that will list each job as running, waiting, failed, or succeeded. 

8. Get the output from HDFS after the jobs are finished.  This command will merge all output into a single text file that can be used by the parallel_bootstrap.py module.
```
hadoop fs -getmerge output_filename output_filename.txt
```