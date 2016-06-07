# -*- coding: utf-8 -*-

from sklearn.cross_decomposition import CCA
import numpy as np
from mvpa2.tutorial_suite import Dataset
import fmri_preprocessing as fp
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from fastdtw import fastdtw
import rcca
from mvpa2.misc.stats import ttest_1samp as ttest

"""
Description: 

    This module contains functions which take as an input a single parameter:
    a Dataset object with shape (s, v, t) where 
            s is the number of subjects
            v is the number of voxels in each subject
            t is the constant number of time series for each voxel
    Each of these functions computes all possible pairwise combinations of a certain 
    metric and returns either a single value or an array of values depending on the metric.
"""


def pearsons_average(ds):
    return 1 - np.mean( pdist(np.mean(ds.samples, axis=1), metric='correlation') )

# Returns the average first canonical correlation, mean centering at each time point  
def cca(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    cca.train([subj.T for subj in centered_ds]) 
    return np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
  
  
# Returns the average first canonical correlation, without mean centering at each time point    
def cca_uncentered(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    cca.train([subj.T for subj in ds.samples]) 
    return np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
    
    
# Returns the p values for a null hypothesis of mean=0 and alternative being mean>0.
# Sample size is the number of subjects in the Dataset. 
def pvalues(ds):
    return ttest(ds.samples.mean(axis=1), popmean=0, alternative='greater')[1]


# Returns the t statistics for a null hypothesis of mean=0 and alternative being mean>0.
# Sample size is the number of subjects in the Dataset.     
def tvalues(ds):  
    return ttest(ds.samples.mean(axis=1), popmean=0, alternative='greater')[0]    


# Returns the average Dynamic Time Warped distance between all pairs.
def dtw_average(ds):	
    X = np.mean(ds.samples, axis=1)
    return np.mean(pdist(X, lambda u, v: fastdtw(u, v)[0]))    
        
    
# Validate with an 80/20 split of training using calculated weights to predict last 
# third of the data and then calculate the mean correlation coefficient  
def cca_validate_predict(ds):
    num_subj = ds.shape[0]
    num_samples = ds.shape[2]
    split_point = int(num_samples * .8)    
    
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    
    train_set = [subj.T[:split_point,:] for subj in centered_ds]
    test_set = [subj.T[split_point:,:] for subj in centered_ds]
            
    cca.train(train_set)    
    weights = np.squeeze(cca.ws, axis=(2,))    
    
    mean_corrs = []
    for i, subj in enumerate(test_set):
        X = np.dot(subj, weights[i].T)        
        corrs = []
        for row in subj.T:            
            corrs.append(np.corrcoef(X, row)[0,1])
        mean_corrs.append(np.mean(corrs))
   
    return mean_corrs
    

# Validate with an 80/20 split of training testing on samples within a subject    
def cca_validate(ds):
    num_subj = ds.shape[0]
    num_samples = ds.shape[2]
    split_point = int(num_samples * .66)
    
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    
    train_set = [subj.T[:split_point,:] for subj in centered_ds]
    test_set = [subj.T[split_point:,:] for subj in centered_ds]

    cca.train(train_set)
    cca.validate(test_set)
    
    cancorr = np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
    predcorr = np.mean(cca.corrs)
    return np.array([cancorr, predcorr])
    