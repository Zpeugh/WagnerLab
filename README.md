# WagnerLab
---
This repository contains functions and methods to analyze fMRI subject movie data.  Specifically, the focus is on
multivariate searchlight driven approaches for intersubject correlations and other measures through time. 

## Directory Descriptions

### wagner_2010/
&nbsp;&nbsp;&nbsp;&nbsp;All of the results and scripts applicable to Dr. Dylan Wagner's 2010 study on the movie <em>Matchstick Men</em>

##### load_data.py
&nbsp;&nbsp;&nbsp;&nbsp;This is a convenience file with methods to load in the 2010 data.

##### tests.py
&nbsp;&nbsp;&nbsp;&nbsp;Various adjustable tests for running on the 2010 dataset

##### results/data/
&nbsp;&nbsp;&nbsp;&nbsp;Pickle files of data small enough to save to GitHub
##### results/figures/ 
&nbsp;&nbsp;&nbsp;&nbsp;Graphs and plots of some results
##### results/nifti/ 
&nbsp;&nbsp;&nbsp;&nbsp;NIFTI files of results

-
### fmri/
&nbsp;&nbsp;&nbsp;&nbsp;The general utility scripts with methods for pre and post processing fMRI timeseries data

##### fmri_preprocessing.py
&nbsp;&nbsp;&nbsp;&nbsp;Here are all of the functions for combining, splicing, and detrending data raw NIFTI data from the fMRI. 

##### dataset_utilities.py
&nbsp;&nbsp;&nbsp;&nbsp;This module holds all of the functions to run analyses on preprocessed Datasets

##### measures.py
&nbsp;&nbsp;&nbsp;&nbsp;All of the methods in this file are used in searchlight analyses

##### rcca.py
&nbsp;&nbsp;&nbsp;&nbsp;This is the source code from [pyrcca](https://github.com/gallantlab/pyrcca/blob/master/rcca.py) 

##### masks/
&nbsp;&nbsp;&nbsp;&nbsp;Directory with NIFTI masks

