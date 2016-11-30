# -*- coding: utf-8 -*-

from sklearn.cross_decomposition import CCA
import numpy as np
from mvpa2.tutorial_suite import Dataset
import fmri_preprocessing as fp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import rcca
from mvpa2.suite import SVM as SVM
from mvpa2.suite import CrossValidation as CV
from mvpa2.suite import NFoldPartitioner as NFoldPartitioner
from mvpa2.misc.stats import ttest_1samp as ttest

"""
Description: 

    This module contains functions which take as an input a single parameter:
    a Dataset object with shape (s, v, t) where 
            s is the number of subjects
            v is the number of voxels in each subject
            t is the constant number of time series for each voxel
    Each of these functions computes a certain metric and returns either a 
    single value or an array of values depending on the metric.
"""

# returns the average Pearson's correlation between all pairs
def pearsons_average(ds):
    return 1 - np.mean( pdist(np.mean(ds.samples, axis=1), metric='correlation') )
    
# return all combinations of Pearson's correlations between pairs
def all_pearsons_averages(ds):
    return 1 - pdist(np.mean(ds.samples, axis=1), metric='correlation')

# Returns all first canonical correlations, pre-mean centered at each time point  
def all_cca(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    cca.train([subj.T for subj in centered_ds]) 
    return cca.cancorrs[0][np.triu_indices(num_subj,k=1)]


# Returns the average first canonical correlation, mean centering at each time point  
def cca(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    cca.train([subj.T for subj in centered_ds])
    if (num_subj == 2):
        return cca.cancorrs[0]
    else:
        return np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
        
# Returns the average first canonical correlation, without mean centering at each time point    
def cca_uncentered(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    cca.train([subj.T for subj in ds.samples]) 
    if (num_subj == 2):
        return cca.cancorrs[0]
    else:    
        return np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
    

# The first subject in the dataset is the one which will be compared to all other subjects
def cca_one_to_all(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    cca.train([subj.T for subj in centered_ds])
    if (num_subj == 2):
        return cca.cancorrs[0]
    else:
        return np.mean(cca.cancorrs[0][0][1:])
        
# Returns the p values for a null hypothesis of mean=0 and alternative being mean>0.
# Sample size is the number of subjects in the Dataset. 
def pvalues(ds):
    return ttest(ds.samples.mean(axis=1), popmean=0, alternative='greater')[1]


# Returns the t statistics for a null hypothesis of mean=0 and alternative being mean>0.
# Sample size is the number of subjects in the Dataset.     
def tvalues(ds):  
    return ttest(ds.samples.mean(axis=1), popmean=0, alternative='greater')[0]    


# Run pyrcca's validate with a 50/50 split of training testing on samples within subjects    
def rcca_validate(ds):
    num_subj = ds.shape[0]
    num_samples = ds.shape[2]
    split_point = int(num_samples * .5)
    
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    
    train_set = [subj.T[:split_point,:] for subj in centered_ds]
    test_set = [subj.T[split_point:,:] for subj in centered_ds]

    cca.train(train_set)
    return np.mean(cca.validate(test_set))

# Run pyrcca's validate with a 50/50 split of training testing on samples within subjects
# Return the maximum correlation from the validation set.    
def rcca_validate_max(ds):
    num_subj = ds.shape[0]
    num_samples = ds.shape[2]
    split_point = int(num_samples * .5)
    
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    
    train_set = [subj.T[:split_point,:] for subj in centered_ds]
    test_set = [subj.T[split_point:,:] for subj in centered_ds]

    cca.train(train_set)
  
    return np.max(np.mean(cca.validate(test_set), axis=0))
    
# Return the mean correlation of all of the correlation matricies for each timepoint
# in a subject.  For example, if each subject has 500 timepoints with 25 voxels at each,
# a correlation matrix of upper_triangle((500x500)) will be stored for each subject.
# Then, all of these correlations will be pairwise correlated to all other subjects
# in a second-level pearsons correlation analysis.  The resulting correlations are
# averaged and returned as a scalar value.
def timepoint_double_corr(ds):
    
    self_correlations = []
    
    for subj in ds.samples:
        corrs = 1 - pdist(subj.T, metric='correlation')
        self_correlations.append(corrs)
        
    correlation = 1 - pdist(self_correlations, metric="correlation")
    
    return np.mean(correlation)
    
    

# Dataset must have cds.a["scene_changes"] Dataset attribute set as an array of integers
# These integers will act as the scene boundaries and must be between 1 and number of 
# samples.  This analysis is the same as the timepoint_isc metric, returning a second
# level average correlation, only using averaged scene activations instead of every 
# timepoint. 
def scene_based_double_corr(ds):
    
    num_subj = ds.shape[0]
    num_voxels = ds.shape[1]
    num_scenes = len(ds.a.scene_changes)
    ds_list = np.zeros((num_subj, num_voxels, num_scenes-1))
    prev_cutoff = 0

    # average correlations for each scene
    for i, scene_cutoff in enumerate(ds.a.scene_changes):
        ds_list[:,:,i] = np.mean(ds.samples[:,:,prev_cutoff:scene_cutoff], axis=2)
        prev_cutoff = scene_cutoff

    self_correlations = []

    # convert each subject to a vector of its pairwise correlations between scenes
    for subj in ds_list:
        corrs = 1 - pdist(subj.T, metric='correlation')
        self_correlations.append(corrs)
    
    # get all pairwise correlations between subjects    
    correlation = 1 - pdist(self_correlations, metric="correlation")

    # return the average isc scene based correlation
    return np.mean(correlation)


# Dataset must have cds.a["scene_changes"] Dataset attribute set as an array of integers
# These integers will act as the scene boundaries and must be between 1 and number of 
# samples. This metric builds an SVM, using averaged scenes as the classes to predict
# N-fold (where N is cds.shape[0] -1, i.e. number of subjects) cross-validation
# is conducted, and the average prediction accuracy of the SVM is returned.
def scene_svm_cross_validation(cds):  
    
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    num_scenes = len(cds.a.scene_changes) - 1
    scenes = cds.a.scene_changes
    ds_list = np.zeros((num_subj, num_voxels, num_scenes))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_scenes - 1):       
       ds_list[:,:,i] = np.mean(cds.samples[:,:,scenes[i]:scenes[i+1]], axis=2)
       
    for subj in ds_list:
        ds_tup = ds_tup + (subj.T, )
        
    ds = Dataset(np.concatenate(ds_tup))  

    ds.sa['subjects'] = np.repeat(np.arange(num_subj), num_scenes)
    ds.sa['targets'] = np.tile(np.arange(num_scenes), num_subj)
    ds.sa['chunks'] = np.tile(np.arange(num_scenes), num_subj)

    clf = SVM()
       
    cv = CV(clf, NFoldPartitioner(attr='subjects'))
    cv_results = cv(ds)
    return 1 - np.mean(cv_results)



# Dataset must have cds.a["scene_changes"] Dataset attribute set as an array of integers
# These integers will act as the scene boundaries and must be between 1 and number of 
# samples. This metric builds an SVM, using averaged scenes as the classes to predict
# N-fold (where N is cds.shape[0] -1, i.e. number of subjects) cross-validation
# is conducted, and a flattened confusion matrix of the results are returned.  
def scene_svm_cross_validation_confusion_matrix(cds):  
    
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    scenes = cds.a.scene_changes
    num_scenes = len(scenes)
    ds_list = np.zeros((num_subj, num_voxels, num_scenes-1))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_scenes - 1):
        if scenes[i] <= scenes[i+1]:
            ds_list[:,:,i] = np.mean(cds.samples[:,:,scenes[i]:scenes[i+1]], axis=2)
        elif scenes[i-1] - scenes[i+1] > 1:
            ds_list[:,:,i] = np.mean(cds.samples[:,:,scenes[i-1]:scenes[i+1]], axis=2)

       
    for subj in ds_list:
        ds_tup = ds_tup + (subj.T, )
        
    ds = Dataset(np.concatenate(ds_tup))  

    ds.sa['subjects'] = np.repeat(np.arange(num_subj), num_scenes)
    ds.sa['targets'] = np.tile(np.arange(num_scenes), num_subj)
    ds.sa['chunks'] = np.tile(np.arange(num_scenes), num_subj)

    clf = SVM()
       
    cv = CV(clf, NFoldPartitioner(attr='subjects'), enable_ca=['stats'])
    cv_results = cv(ds)
    return cv.ca.stats.matrix.flatten()

# Dataset must have cds.a["scene_changes"] Dataset attribute set as an array of integers
# These integers will act as the scene boundaries and must be between 1 and number of 
# samples. Additionally, cds.a["clusters_per_iter"] must be set as an integer number
# between 1 and the (<total number of timepoints> / <number of scenes>), with somewhere 
# around 1/8th the number of timepoints being a good compromise for speed and accuracy.  
# This algorithm hierarchically clusters timeseries into the same number of scenes as
# the scene boundaries given using pearson's correlation.  Then the average correlation 
# of the artificially created clustered scenes and the actual given scenes will be 
# returned as a scalar 
def cluster_scenes(cds):
    
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    scenes = cds.a.scene_changes
    clusters_per_iter = cds.a.clusters_per_iter
    n_scenes = len(scenes)
    iteration = 0
    
    samples = np.mean(cds.samples, axis=0).T
        
    n_samples = samples.shape[0]
    n_clusters = n_samples
    
    while len(samples) > n_scenes:              
        iteration += 1        
        correlations = []
        new_cluster = []
        clusters = []
        last_merged = -1   
        if len(samples) - clusters_per_iter < n_scenes:
            clusters_per_iter = len(samples) - n_scenes
        for i in range(len(samples) - 1):
            correlations.append(np.corrcoef(samples[i], samples[i+1])[0,1])
        
        corrs = np.array(correlations)
        max_indices = np.argpartition(corrs, -clusters_per_iter)[-clusters_per_iter:]

        for j in range(len(samples) - 1):
            if j in max_indices:
                if j-1 in max_indices:
                    last_merged = j+1
                    new_cluster = np.mean([clusters[-1], samples[j+1]], axis=0)
                    clusters[-1] = new_cluster                    
                else:
                    last_merged = j + 1
                    new_cluster = np.mean([samples[j], samples[j+1]], axis=0)                   
                    clusters.append(new_cluster)                    
            elif j != last_merged:                    
                clusters.append(samples[j])
        if last_merged is not len(samples) - 1:
            clusters.append(samples[len(samples) - 1])
        samples = clusters       
    
    samples = np.array(samples)

    ################### compare with scene boundaries given ##################    
    
    ds_list = np.zeros((n_scenes, num_voxels))
    
    prev_cutoff = 0
    scene_samples = np.mean(cds.samples, axis=0).T
    
    # average correlations for each scene
    for i, scene_cutoff in enumerate(cds.a.scene_changes):
        ds_list[i,:] = np.mean(scene_samples[prev_cutoff:scene_cutoff,:], axis=0)
        prev_cutoff = scene_cutoff
    
    return np.mean([np.corrcoef(samples[i],ds_list[i])[0,1] for i in range(n_scenes)])
    
# Dataset must have cds.a["scene_changes"] Dataset attribute set as an array of integers
# These integers will act as the scene boundaries and must be between 1 and number of 
# samples. Additionally, cds.a["clusters_per_iter"] must be set as an integer number
# between 1 and the (<total number of timepoints> / <number of scenes>), with somewhere 
# around 1/8th the number of timepoints being a good compromise for speed and accuracy.  
# This algorithm hierarchically clusters timeseries into the same number of scenes as
# the scene boundaries given using pearson's correlation.  Then the euclidean distance
# between the original scene boundary indices and the ones generated by the clustering
# method will be returned.
def cluster_scenes_track_indices(cds):
    
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    scenes = cds.a.scene_changes
    clusters_per_iter = cds.a.clusters_per_iter
    n_scenes = len(scenes)
    iteration = 0
    
    samples = np.mean(cds.samples, axis=0).T
    
    # add the indices as an extra dimension at the beginning of the voxels
      
    n_samples = samples.shape[0]
    samples = np.hstack((np.arange(n_samples).reshape((n_samples,1)), samples))
    
    while len(samples) > n_scenes:                
        iteration += 1        
        correlations = []
        new_cluster = []
        clusters = []
        last_merged = -1   
        if len(samples) - clusters_per_iter < n_scenes:
            clusters_per_iter = len(samples) - n_scenes
        for i in range(len(samples) - 1):
            correlations.append(np.corrcoef(samples[i][1:], samples[i+1][1:])[0,1])
        
        corrs = np.array(correlations)
        max_indices = np.argpartition(corrs, -clusters_per_iter)[-clusters_per_iter:]
        
        for j in range(len(samples) - 1):
            if j in max_indices:
                if j-1 in max_indices:
                    last_merged = j+1
                    old_index = clusters[-1][0]
                    new_index = samples[j+1][0]                    
                    new_cluster = np.mean([clusters[-1], samples[j+1]], axis=0)
                    if new_index > old_index:
                        new_cluster[0] = new_index
                    else:
                        new_cluster[0] = old_index
                    clusters[-1] = new_cluster                    
                else:
                    last_merged = j + 1
                    old_index = samples[j][0]
                    new_index = samples[j+1][0]
                    new_cluster = np.mean([samples[j], samples[j+1]], axis=0)
                    if new_index > old_index:
                        new_cluster[0] = new_index
                    else:
                        new_cluster[0] = old_index
                    clusters.append(new_cluster)                    
            elif j != last_merged:              
                clusters.append(samples[j])
        if last_merged is not len(samples) - 1:
            clusters.append(samples[len(samples) - 1])
        samples = clusters       
    
    
    indices = np.array(samples)[:,0]    
    
    return euclidean(indices, cds.a.scene_changes)