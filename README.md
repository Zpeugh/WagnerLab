# WagnerLab
This repository contains functions and methods to analyze fMRI subject movie data.  Specifically, the focus is on
multivariate searchlight driven approaches for intersubject correlations and other measures through time. 

## Directory Descriptions

>### fmri/wagner_2010/
All of the results and scripts applicable to Dr. Dylan Wagner's 2010 study on the movie <em>Matchstick Men</em>

>>##### load_data.py
>>>This is a convenience file with methods to load in the 2010 data.

>>##### analysis.py
>>>Various adjustable tests for running on the 2010 dataset

>>##### results/data/
>>>Pickle files of data small enough to save to GitHub

>>##### results/figures/ 
>>>Graphs and plots of some results

>>##### results/nifti/ 
>>>NIFTI files of results

-
>### fmri/fmri/
The general utility scripts with methods for pre and post processing fMRI timeseries data

>>##### fmri_preprocessing.py
>>>Here are all of the functions for combining, splicing, and detrending data raw NIFTI data from the fMRI. 

>>##### dataset_utilities.py
>>>This module holds all of the functions to run analyses on preprocessed Datasets

>>##### measures.py
>>>All of the methods in this file are used in searchlight analyses

-
>### masks/
Directory with NIFTI masks

>setup.py
Run this script from this directory to temporarily add all modules from the fmri/fmri directory into your workspace
