# WagnerLab
This repository contains functions and methods to analyze fMRI subject movie data.  Specifically, the focus is on
multivariate searchlight driven approaches for intersubject correlations and other measures through time. 
## File Descriptions


### fmri_preprocessing.py
Here are all of the functions for combining, splicing, and detrending data raw NIFTI data from the fMRI. 

### load_data.py
This is a convenience file with methods to load in the 2010 data.

### dataset_utilities.py
This module holds all of the functions to run analyses on preprocessed Datasets

### measures.py
All of the methods in this file are used in searchlight analyses

### rcca.py
This is the source code from [pyrcca](https://github.com/gallantlab/pyrcca/blob/master/rcca.py) 

### tests.py
Various adjustable tests for running on the 2010 dataset

### masks/
Directory with NIFTI masks

### results/

* data- the pickle files of data small enough to save to GitHub
* figures- Graphs and plots of some results
* nifti- NIFTI files of results

