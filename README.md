# WagnerLab
This repository contains functions and methods to analyze fMRI subject movie data.  Specifically, the focus is on
multivariate searchlight driven approaches for intersubject correlations and other measures through time. 

## Directory Descriptions

###fmri/wagner_2010/ ###
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; All of the results and scripts applicable to Dr. Dylan Wagner's 2010 study on the movie <em>Matchstick Men</em>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>load_data.py</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is a convenience file with methods to load in the 2010 data.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>analysis.py</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Various adjustable tests for running on the 2010 dataset

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>results/data/</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pickle files of data small enough to save to GitHub

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>results/figures/</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Graphs and plots of some results

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>results/nifti/</strong><br />
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; NIFTI files of results</p><br />

### fmri/fmri/ ###
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The general utility scripts with methods for pre and post processing fMRI timeseries data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>fmri_preprocessing.py</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Here are all of the functions for combining, splicing, and detrending data raw NIFTI data from the fMRI. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>dataset_utilities.py</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This module holds all of the functions to run analyses on preprocessed Datasets

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>measures.py</strong><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; All of the methods in this file are used in searchlight analyses<br />

### fmri/masks/ ###
Directory with NIFTI masks

### fmri/setup.py ###
Run this script from this directory to temporarily add all modules from the fmri/fmri directory into your workspace
