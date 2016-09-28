# WagnerLab
This repository contains functions and methods to analyze fMRI subject movie data.  Specifically, the focus is on
multivariate searchlight driven approaches for intersubject correlations and other measures through time. 

## Directory Descriptions

### wagner_2010/
&nbsp;&nbsp;&nbsp;&nbsp;All of the results and scripts applicable to Dr. Dylan Wagner's 2010 study on the movie <em>Matchstick Men</em>

>##### load_data.py
>>This is a convenience file with methods to load in the 2010 data.

>##### tests.py
>>Various adjustable tests for running on the 2010 dataset

>##### results/data/
>>Pickle files of data small enough to save to GitHub

>##### results/figures/ 
>>Graphs and plots of some results

>##### results/nifti/ 
>>NIFTI files of results

-
### fmri/
&nbsp;&nbsp;&nbsp;&nbsp;The general utility scripts with methods for pre and post processing fMRI timeseries data

>##### fmri_preprocessing.py
>>Here are all of the functions for combining, splicing, and detrending data raw NIFTI data from the fMRI. 

>##### dataset_utilities.py
>>This module holds all of the functions to run analyses on preprocessed Datasets

>##### measures.py
>>All of the methods in this file are used in searchlight analyses

>##### rcca.py
>>This is the source code from [pyrcca](https://github.com/gallantlab/pyrcca/blob/master/rcca.py) 

>##### masks/
>>Directory with NIFTI masks

