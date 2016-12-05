# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:09:46 2016

@author: peugh.14
"""

import numpy as np
import matplotlib.pyplot as plt
from mvpa2.tutorial_suite import *
import fmri_preprocessing as fp
from mvpa2.measures.searchlight import sphere_searchlight
import matplotlib.patches as mpatches
from mvpa2.misc.stats import ttest_1samp as ttest
from measures import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy 
from sklearn.manifold import MDS



'''====================================================================================
    Plot the timeseries of a single voxel for a Dataset using matplotlib.
    
    ds                  The Dataset object containing samples
    voxel_position      a number representing which voxel in the dataset to display
======================================================================================'''    
def voxel_plot(ds, voxel_position):
    plt.clf()    
    plt.figure(figsize=(10,6))
    plt.plot(np.transpose(ds.samples)[voxel_position])
    plt.title("Timeseries for voxel {0}".format(voxel_position))
    plt.axvline(211, color='r', linestyle='--')
    plt.axvline(422, color='r', linestyle='--')
    plt.show()


'''====================================================================================
    Plot the intersubject information versus intersubject canonical correlation in
    a single color scatterplot.
    
    isc_data            The Dataset object containing (1, n) intersubject canonical
                        correlation data samples
    isc_data            The Dataset object containing (1, n) intersubject correlation
                        data samples
    title               The string title for the figure
    save                (optional) Boolean variable for whether or not you wish to save
                        the file
    filename            Required if save=True. The name of the file to save the figure as
======================================================================================'''  
def plot_isc_vs_isi(isc_data, isi_data, title, save=False, filename=None):
    plt.clf()    
    fig = plt.figure(figsize=(10,6))
    plt.scatter(isc_data, isi_data)
    plt.title(title)
    plt.xlabel('Intersubject Correlation')
    plt.ylabel('intersubject Information')
    
    if save:
        fig.savefig(filename)
    plt.show()
 

'''====================================================================================
    Plot the intersubject information versus intersubject canonical correlation in
    a tri-colored scatterplot, where blue indicates the back 1/3rd of the brain, green
    indicates the middle 1/3rd and red indicates the front 1/3rd (all with respect to the
    sagittal plane).
    
    isc_data            The Dataset object containing (1, n) intersubject canonical
                        correlation data samples
    isc_data            The Dataset object containing (1, n) intersubject correlation
                        data samples
    title               The string title for the figure
    xlabel              (optional) The label for the x axis.  Defaults to 
                        'Intersubject Correlation'
    ylabel              (optional) The label for the y axis.  Defaults to 
                        'Intersubject Information'
    save                (optional) Boolean variable for whether or not you wish to save
                        the file
    filename            Required if save=True. The name of the file to save the figure as
======================================================================================'''  
def plot_colored_isc_vs_isi(isc_data, isi_data, title, xlabel='Intersubject Correlation', ylabel='Intersubject Information', save=False, filename=None):
   
    voxels = isc_data.fa.voxel_indices    
    
    iscd = isc_data.samples[0,:]
    isid = isi_data.samples[0,:]
    
    blue_x = [x for i, x in enumerate(iscd) if voxels[i][1] <= 20]
    blue_y = [x for i, x in enumerate(isid) if voxels[i][1] <= 20]
    green_x = [x for i, x in enumerate(iscd) if voxels[i][1] > 20 and voxels[i][1] < 41]
    green_y = [x for i, x in enumerate(isid) if voxels[i][1] > 20 and voxels[i][1] < 41]
    red_x = [x for i, x in enumerate(iscd) if voxels[i][1] >= 41]
    red_y = [x for i, x in enumerate(isid) if voxels[i][1] >= 41]
    
    color_array = ['seagreen' for i in range(len(green_x))]
    color_array += ['darkred' for i in range(len(red_x))] 
    color_array += ['steelblue' for i in range(len(blue_x))]       
       
    X = green_x + red_x + blue_x
    Y = green_y + red_y + blue_y

    plt.clf()
    fig = plt.figure(figsize=(10,6))
    blue_marker = mpatches.Patch(color='steelblue', label="back")
    green_marker = mpatches.Patch(color='seagreen', label="middle")
    red_marker = mpatches.Patch(color='darkred', label="front")
    
    plt.scatter(X,Y, marker = 'x', color=color_array)
    fig.legend( handles=[blue_marker, green_marker, red_marker], 
                labels=["Back 1/3rd", "Middle 1/3rd", "Front 1/3rd"], 
                bbox_to_anchor=(0.66,0.125), loc='lower left' )
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    
    if save:
        fig.savefig(filename)
    plt.show()
 
 
'''====================================================================================
    Plot the intersubject information versus intersubject canonical correlation in
    a quad-colored scatterplot.
    
    isc_data            The Dataset object containing (1, n) intersubject canonical
                        correlation data samples
    isc_data            The Dataset object containing (1, n) intersubject correlation
                        data samples
    title               The string title for the figure
    isc_p_thresh        The false discovery threshold value for the intersubject 
                        correlation
    isi_p_thresh        The false discovery threshold value for the intersubject 
                        information (canonical correlation)
    xlabel              (optional) The label for the x axis.  Defaults to 
                        'Intersubject Correlation'
    ylabel              (optional) The label for the y axis.  Defaults to 
                        'Intersubject Information'
    save                (optional) Boolean variable for whether or not you wish to save
                        the file
    scale_axis          (optional) If True the x and Y axis will have the same scale, IF
                        not then they will be scaled automatically
    filename            Required if save=True. The name of the file to save the figure as
======================================================================================''' 
def plot_thresholded_isc_vs_isi(isc_data, isi_data, title, isc_p_thresh, isi_p_thresh, xlabel='Intersubject Correlation', ylabel='Intersubject Information', save=False, filename=None, same_scale_axis=False):
   
    voxels = isc_data.fa.voxel_indices    
    
    iscd = isc_data.samples[0,:]
    isid = isi_data.samples[0,:]
    
    blue_x =  [x for i, x in enumerate(iscd) if x >= isc_p_thresh and isid[i] < isi_p_thresh]
    blue_y =  [x for i, x in enumerate(isid) if x < isi_p_thresh and iscd[i] >= isc_p_thresh]
    green_x = [x for i, x in enumerate(iscd) if x < isc_p_thresh and isid[i] >= isi_p_thresh] 
    green_y = [x for i, x in enumerate(isid) if x >= isi_p_thresh and iscd[i] < isc_p_thresh] 
    red_x =   [x for i, x in enumerate(iscd) if x >= isc_p_thresh and isid[i] >= isi_p_thresh]
    red_y =   [x for i, x in enumerate(isid) if x >= isi_p_thresh and iscd[i] >= isc_p_thresh]
    black_x = [x for i, x in enumerate(iscd) if x < isc_p_thresh and isid[i] < isi_p_thresh]
    black_y = [x for i, x in enumerate(isid) if x < isi_p_thresh and iscd[i] < isc_p_thresh]
    
    color_array = ['steelblue' for i in range(len(blue_x))]
    color_array += ['seagreen' for i in range(len(green_x))]
    color_array += ['darkred' for i in range(len(red_x))] 
    color_array += ['black' for i in range(len(black_x))]       
       
    X = blue_x + green_x + red_x + black_x
    Y = blue_y + green_y + red_y + black_y


    plt.clf()
    fig = plt.figure(figsize=(10,6))
    blue_marker = mpatches.Patch(color='steelblue', label="ISC only")
    green_marker = mpatches.Patch(color='seagreen', label="ISI only")    
    red_marker = mpatches.Patch(color='darkred', label="Both")
    black_marker = mpatches.Patch(color='black', label="Neither")
    
    plt.scatter(X,Y, marker = '.', color=color_array)
    plt.axvline(isc_p_thresh, color="black", linewidth=2)
    plt.axhline(isi_p_thresh, color="black", linewidth=2)
    fig.legend( handles=[blue_marker, green_marker, red_marker, black_marker], 
                labels=["ISC only", "ISI only", "Both", "Neither"], 
                bbox_to_anchor=(0.66,0.125), loc='lower left' )
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    
    if same_scale_axis:   
        ax_min = np.min(isi_data) - 0.05
        ax_max = np.max(isi_data) + 0.05
        plt.axis([-0.05, ax_max, -0.05, ax_max])
    
    if save:
        fig.savefig(filename)
    plt.show() 
 

'''====================================================================================
    Plot the timeseries of number of voxels activated above a significance threshold.
    
    ds          The Dataset object containing pvalues testing the null
                hypothesis of mean=0.  
    a           The alpha level to count as 'activated'
    n           The number of subjects in the test
    filename    If this is set then the plot will be saved, otherwise it will just 
                be displayed
======================================================================================'''
def plot_significant(ds, a=0.05, n=34, filename=None):
    X = ds.samples
    min_t = scipy.stats.t.ppf(1-a, n)    
    U = np.ma.masked_greater(X, min_t).mask  
    
    sums = [np.sum(x) for x in U]
    
    plt.clf()
    fig = plt.figure(figsize=(10,6))
    plt.plot(sums)
    plt.xlabel('Time (2.5s increments)')
    plt.ylabel('Total Voxels Activated')
    plt.title('Voxel Activation Significantly Above 0 With a={0}'.format(a))
    if filename:
       fig.savefig(filename) 
    plt.show()
    return np.array(sums)

  

'''====================================================================================
    Takes a dataset with sample attributes of features and returns an ordered list of
    all possible pairwise combinations either multiplied, divided, or subtracted.  The 
    order returned is that same as that which scipy.spatial.distance.pdist returns.
    
    ds          The Dataset object with sample attributes to pairwise combine.
    feature     The feature in ds.sa
    method      One of 'multiply', 'divide', 'difference'
    
    returns     a list of length (n choose 2) where n is the length of the feature array
======================================================================================''' 
def pairwise_feature_list(ds, feature, method='multiply'):
    feature = ds.sa[feature]
    num_subj = len(feature)
    X = []
    for i in range(num_subj - 1):
        for j in range(i+1, num_subj):
            if method == 'multiply':
                X.append(feature[i] * feature[j])
            elif method == 'difference':
                X.append(abs(feature[i] - feature[j]))
            elif method == 'divide':
                X.append(feature[i] / feature[j])
    return X


'''====================================================================================
    Plot an (n, m) design matrix in grayscale using matplotlib
    
    dm          The numpy (n,m) array where n is the number of samples and m is the 
                number of variables being regressed out.    
======================================================================================'''    
def show_design_matrix(dm):
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.imshow(dm, cmap='gray', aspect='auto', interpolation='none')
    plt.show()


'''====================================================================================
    Export a dataset to an NiFti file.
    
    ds          The Dataset object to be exported
    filename    The string of the filename    
======================================================================================'''    
def export_to_nifti(ds, filename):    
    img = map2nifti(ds)
    img.to_filename(filename)


'''====================================================================================
    Takes n datasets with v voxels and t time samples each, and creates a numpy array 
    with shape (n, v, t) and inserts it into a new Dataset object with the same voxel
    indices as before. 
    
    ds_list      The list of Dataset objects containing subject data. 
                        
    Returns      A new Dataset object with all of the datasets combined
======================================================================================'''   
def combine_datasets(dslist, transpose=True):
    num_samples = dslist[0].shape[0]
    num_voxels = dslist[0].shape[1]
    ds_tup = ()
    for subj in dslist:
        if transpose:
            ds_tup = ds_tup + (subj.samples.T,)
        else:
            ds_tup = ds_tup + (subj.samples,)
    ds = Dataset(np.vstack(ds_tup).reshape((len(dslist), num_voxels, num_samples)))
    ds.a.mapper = dslist[0].a.mapper
    ds.fa["voxel_indices"] = dslist[0].fa.voxel_indices
    ds.sa.clear()
    ds.sa["subject"] = np.arange(len(dslist))       
    return ds


    
'''====================================================================================
    Takes a dataset of shape (n, v, t) where n is number of subjects, v is number
    of voxels, and t is number of time samples for each subject.  Runs a parallel
    searchlight analysis on all of the subjects given the metric input.
    
    ds          The Dataset object containing all of the subjects runs
    metric      (optional) A string representing one of the measures in the measures 
                module, or alternatively, any method which takes a single parameter of
                a Dataset and returns either a scalar or a 1 dimensional array of scalars
                Defaults to Pearsons Correlation. 
    radius      (optional) The radius of the searchlight sphere. Defaults to 2.
    center_ids  (optional) The feature attribute name to use as the centers for the 
                searchlights.
    nproc       (optional) Number of processors to use.  Defaults to all available 
                processors on the system. 
                        
    Returns      The results of the searchlight analysis
======================================================================================''' 
def run_searchlight(ds, metric='correlation', radius=2, center_ids=None, n_cpu=None):

    if metric == 'cca_u':
        measure = cca_uncentered
    elif metric == 'cca':
        measure = cca
    elif metric == '1_to_many_cca':
        measure = cca_one_to_all
    elif metric == 'all_cca':
        measure = all_cca
    elif metric == 'cca_validate':
        measure = cca_validate
    elif metric == 'cca_validate_max':
        measure = cca_validate_max
    elif metric == 'correlation':
        measure = pearsons_average
    elif metric == 'all_pearsons':
        measure = all_pearsons_averages
    elif metric == 'pvalues':
        measure = pvalues
    elif metric == 'tvalues':
        measure = tvalues
    elif metric == "timepoint_double_corr":
        measure = timepoint_double_corr
    elif metric == "scene_based_double_corr":
        measure = scene_based_double_corr
    elif metric == "scene_svm_cv":
        measure = scene_svm_cross_validation
    elif metric =="scene_svm_cv_cm":
        measure = scene_svm_cross_validation_confusion_matrix
    elif metric == "cluster_scenes":
        measure = cluster_scenes
    elif metric == "cluster_scenes_track_indices":
        measure = cluster_scenes_track_indices
    elif metric == "cluster_scenes_return_indices":
        measure = cluster_scenes_return_indices
    else:
        measure = metric
        
    sl = sphere_searchlight(measure, radius=radius, center_ids=center_ids, nproc=n_cpu)   
        
    searched_ds = sl(ds)
    searched_ds.fa = ds.fa
    searched_ds.a = ds.a    
    
    return searched_ds


'''====================================================================================
    Rolls data a random number of permutations on the longest axis.  For example,
    the array [1,2,3,4,5,6] could be returned as something like [4,5,6,1,2,3].  The axis
    with the highest dimensionality is used to roll on. 
    
    ds      the arraylike object of shape (n, m) with time points to randomly shift
    
    returns the shifted array
======================================================================================''' 
def shift_subject(ds):
    
    num_samples = 1
    min_perm = 1
    
    if ds.shape[0] < ds.shape[1]:
        num_samples = ds.shape[1]
        axis = 1
    else:
        num_samples = ds.shape[0]
        axis = 0
        
    if num_samples > 10:
        min_perm = 10
    
    rand_perm = np.random.randint(min_perm, num_samples-min_perm)
    
    return np.roll(ds, rand_perm, axis=axis)
    
    
    
'''====================================================================================
    Returns a completely randomly shuffled version of a dataset with shape (n, m)
    
    ds      dataset with shape (n, m)
    
    returns the shuffled dataset
======================================================================================''' 
def randomize_subject(ds):
    ds_copy = ds.copy()
    total_samples = ds.shape[0] * ds.shape[1]
    ds_copy = ds_copy.reshape(total_samples)
    np.random.shuffle(ds_copy)
    return ds_copy.reshape((ds.shape[0], ds.shape[1]))
    
    
'''====================================================================================
    Takes a dataset with confusion matrices at each voxel in the brain and computes 
    either the average accuracy accross all labels, or the accuracy for each label 
    returning either a scalar or a vector depending on the parameter "average"    
    
    ds       dataset containing an array of confusion matrices
    average  (optional) Defaults to False.  If True, then the average of all of the 
             confusion matrices is returned as a single scalar
    
    returns  an array or averaged scalar accuracy of the confusion matrics
======================================================================================''' 
def confusion_matrix_accuracies(ds, average=False):
    
    result = []
    for mat in ds:
        result.append(np.true_divide(mat.diagonal(), mat.sum(axis=0)))    
    if average:
        return result.mean(axis=1)
    else:    
        return result
     
'''====================================================================================
    Takes a Dataset object and performs hierarchical clustering using Pearson's
    correlation as the metric.  The clusters are created on the time axis.
    
    cds       the Dataset object with shape (s, v, t) where 
                  s is the number of subjects
                  v is the number of voxels in each subject
                  t is the constant number of time series for each voxel  
    scenes    (optional) It is assumed that the original cluster boundaries to average
              into are in the dataset attribute cds.a["event_bounds"].  However, if this
              is not the case, users can supply their own cluster boundaries by passing
              an array for this value.  For example, passing
              clusters= np.arange(cds.samples.shape[2]) would cluster beginning with all
              timepoints treated as their own cluster.
             
    filename  If this is not None, then the dendrogram is saved to disk.
    
    displays  the dendrogram results of the clustering.
======================================================================================'''
def create_dendrogram(cds, clusters=None, filename=None):    
    
    
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    
    if clusters == None:
        clusters = cds.a.event_bounds
        
    num_scenes = len(clusters)
    ds_list = np.zeros((num_subj, num_voxels, num_scenes-1))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_scenes - 1):
        ds_list[:,:,i] = np.mean(cds.samples[:,:,clusters[i]:clusters[i+1]], axis=2)
       
    Z = hierarchy.linkage(np.mean(ds_list, axis=0).T, metric='correlation')
        
    fig = plt.figure(figsize=(14,8))
    hierarchy.dendrogram(Z)
    plt.show()
    if filename is not None:
        fig.savefig(filename)
        
        
'''====================================================================================
    Takes a Dataset object and Multidimensional Scaling using Pearson's
    correlation as the metric.
    
    cds       the Dataset object with shape (s, v, t) where 
                  s is the number of subjects
                  v is the number of voxels in each subject
                  t is the constant number of time series for each voxel  
    scenes    (optional) It is assumed that the original cluster boundaries to average
              into are in the dataset attribute cds.a["event_bounds"].  However, if this
              is not the case, users can supply their own cluster boundaries by passing
              an array for this value.  For example, passing
              clusters= np.arange(cds.samples.shape[2]) would cluster beginning with all
              timepoints treated as their own cluster.
             
    filename  If this is not None, then the dendrogram is saved to disk.
    
    displays  the mds results
======================================================================================'''       
def clustered_mds(cds, clusters=None, filename=None):    

    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    clusters = cds.a.event_bounds
    num_clusters = len(clusters)
    ds_list = np.zeros((num_subj, num_voxels, num_clusters-1))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_clusters - 1):
        ds_list[:,:,i] = np.mean(cds.samples[:,:,clusters[i]:clusters[i+1]], axis=2)
       
    dsm_array = []    
    for subj in ds_list:        
        dsm_array.append(squareform(1 - pdist(subj.T, metric='correlation')))
        
    dsm = np.mean(dsm_array, axis=0)
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    coords = mds.fit(dsm).embedding_
    
    plt.clf()
    X, Y = coords[:,0], coords[:,1]
    labels = np.arange(1,num_clusters)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    plt.scatter(X,Y, marker='x')
    for i, label in enumerate(np.arange(1,num_clusters)):
        ax.annotate(label, (X[i],Y[i]))    
        
    plt.axis([np.min(X)*1.2, np.max(X)*1.2, np.min(Y)*1.2, np.max(Y)*1.2])
    plt.title("MDS Scene Visualization")
    plt.show()
    
    return dsm