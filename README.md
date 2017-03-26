# parallel_bootstrap

Use bootstrapping to estimate the sampling distribution of model fit statistics and parameter estimates using cluster computing.

### I. Introduction

Bootstrapping is a powerful technique to estimate the sampling distribution of a statistic by resampling from the empirical distribution.  We sample from the the observed data with replacement, fit a model to the resulting sample, and compute the statistic of interest based on the fit (e.g., parameter estimates, goodness-of-fit statistic, etc).  This provides several crucial advantages including (1) quantifying uncertainty about paramter estimates, (2) testing for significant improvements in fit across models, and (3) aggregating model predictions over samples for more stable estimates (i.e., bootstrap aggregation or "bagging").  The primary disadvantage is that it can be computationally demanding.  Fortunately, because each sampling and fitting process is independent, this procedure ideal for paralellization.

This code implements parallel bootstrapping using cluster computing.  The user provides a feature matrix (observations x features) and a column vector containing the target variable in the form of .csv files.  In parallel across computing nodes, the data are sampled and fitted using linear regression models with L1 or L2 regularization.  The best-fit parameters and some summary statistics are compiled in an output file.  Code is provided to format the resulting output for subsquent analyses in Python.

This code was designed to run on [NYU's High-Performance Computing (HPC) cluster](https://wikis.nyu.edu/display/NYUHPC/High+Performance+Computing+at+NYU).  Two sets of code are provided for two different clusters that use different resource management software.  The **sparkBoot** directory can be used on Dumbo, the 48-node Hadoop cluster, running Cloudera CDH 5.9.0 (Hadoop 2.6.0 with Yarn).  It makes use of Apache Spark, a software library that operates on Hadoop data, and it's Python API, PySpark.  The **bashBoot** directory can be used on Mercer, the 394-node, 5812-core, primary cluster.  Mercer uses Torque for resource managment.  Bash scripts are provided to automate distribution of jobs across nodes.  The code has only been tested on these clusters, but should work on clusters running the same management software.  See the README files in the sparkBoot and bashBoot directories for more details.  


### II. Preparing the data

The model features (independent variables) should be saved in an observations x features matrix in csv format (e.g., using the pandas to_csv function in Python).  The target variable (dependent variable) should be saved as an observations x 1 column vector in csv format.  Both csv files should be placed inside the /data subdirectory of the sparkBoot or bashBoot folder.  Examples are included in the /data subdirectories.

### III. Run sparkBoot or bashBoot

Move the sparkBoot or bashBoot directories to your user directory on the cluster.  The output will be saved in a txt file that includes fit statistics for each bootstrap sample.  The output is identical regardless of which code and cluster is used.  See the README.md files in each subdirectory for specific instructions.

### IV.  Analyze the output.

The parallel_bootstrap module includes a set of functions for analyzing output from sparkBoot and bashBoot.
- txt2df:    Convert the output of bashBoot and sparkBoot to a pandas dataframe for subsequent analysis.
- predict:   Given a set of observed features, predict the target variable using each bootstrap fit.
- rmse_dist: Compute root mean squared error for each bootstrapped sample.
- r2_dist: Compute coefficient of determination (r-squared) for each bootstrapped sample.

A notebook, parallel_bootstrap_analysis.ipynb, demonstrates the use of these functions.  Example output is provided to run that can be used with the notebook in the /example_output directory.

### V. Potential future improvements

- Better way to handle user input than command line arguments
- More options for models (e.g., SVR, decision trees)
- Utilizing parallel cross-validation for each job.  At present, each job utilizes only a single job.






