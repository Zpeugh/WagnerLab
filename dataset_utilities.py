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

    voxel_position      a number representing which voxel in the dataset to display
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
    save                (optional) Boolean variable for whether or not you wish to save
                        the file
    filename            Required if save=True. The name of the file to save the figure as

    voxel_position      a number representing which voxel in the dataset to display
======================================================================================'''  
def plot_colored_isc_vs_isi(isc_data, isi_data, title, save=False, filename=None):
   
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
    plt.xlabel('Intersubject Correlation',fontsize=15)
    plt.ylabel('Intersubject Information', fontsize=15)
    
    if save:
        fig.savefig(filename)
    plt.show()
 

'''====================================================================================
    Plot the timeseries of number of voxels activated above a significance threshold.
    
    ds          The Dataset object containing pvalues testing the null
                hypothesis of mean=0.  
    a           The alpha level to count as 'activated'
    n           The number of subjects in the test
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

# Work in progress.
def plot_scenes(ds, a, filename=None):
    
    FALSE_DISCOVERY_RATE = 0.05
    TOTAL_VOXELS_IN_BRAIN = ds.shape[1]
    X = ds.samples
    U = np.ma.masked_inside(X, 0,a).mask
    
    sums = [np.sum(x) for x in U]    
    threshold = FALSE_DISCOVERY_RATE * TOTAL_VOXELS_IN_BRAIN
    ones = np.zeros(ds.shape[0])
    
    for i, s in enumerate(sums):
        if s > threshold:
            ones[i] = 1
            
    plt.clf()
    fig = plt.figure(figsize=(10,6))
    plt.plot(ones, '.')
    plt.xlabel('Time (2.5s increments)')
    plt.ylabel('Activated (T/F)')
    plt.ylim([0, 1.5])
    plt.title('Binary Activations')
    if filename:
       fig.savefig(filename) 
    plt.show()       

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
    metric      (optional) One of 'euclidean', 'dtw', 'cca', 'correlation', 'tvalues',
                'pvalues', 'cca_vp'.
                Defaults to Pearsons Correlation. 
    radius      (optional) The radius of the searchlight sphere. Defaults to 2.
    center_ids  (optional) The feature attribute name to use as the centers for the 
                searchlights.
    nproc       (optional) Number of processors to use.  Defaults to all available 
                processors on the system. 
                        
    Returns      The results of the searchlight
======================================================================================''' 
def run_searchlight(ds, metric='correlation', radius=2, center_ids=None, n_cpu=None):

    if metric == 'dtw':
        measure = dtw_average
    elif metric == 'cca_u':
        measure = cca_uncentered
    elif metric == 'cca':
        measure = cca
    elif metric == 'all_cca':
        measure = all_cca
    elif metric == 'cca_validate':
        measure = cca_validate
    elif metric == 'rcca_validate':
        measure = rcca_validate
    elif metric == 'correlation':
        measure = pearsons_average
    elif metric == 'all_pearsons':
        measure = all_pearsons_averages
    elif metric == 'pvalues':
        measure = pvalues
    elif metric == 'tvalues':
        measure = tvalues
    else:
        print("Invalid metric, using Pearson's Correlation by default.")
        measure = pearsons_average
        
    sl = sphere_searchlight(measure, radius=radius, center_ids=center_ids, nproc=n_cpu)   
        
    searched_ds = sl(ds)
    searched_ds.fa = ds.fa
    searched_ds.a = ds.a    
    
    return searched_ds


'''====================================================================================
TODO:  Finish this function so that it returns a neat analysis of a time segment. 
======================================================================================'''   
def segment_analysis(ds, t_start, t_end, metric='all', radius=2, n_cpu=20): 
   
    split_ds = Dataset(ds.samples[:, :, t_start:t_end])   
    split_ds.fa = ds.fa
    
    if metric == 'all':
        corr_res = run_searchlight(split_ds, metric='correlation', radius=radius, n_cpu=n_cpu)
        print("\nThe average correlation is: {0}".format(np.mean(corr_res.samples)))
        print("The maximum correlation is: {0}".format(np.max(corr_res.samples)))        
        
        t_res = run_searchlight(split_ds, metric='tvalues', radius=radius, n_cpu=n_cpu)       
        print("\nThe average t-value is: {0}".format(np.mean(t_res.samples)))
        print("The maximum t-value is: {0}".format(np.max(t_res.samples)))
        return corr_res, t_res
    else:
        return run_searchlight(split_ds, metric=metric, radius=radius, n_cpu=n_cpu)
   

def scene_segmentation_analysis(cds, scenes, metric='correlation', radius=2, n_cpu=20):
    scene_correlations = dict()
    scene_ds = dict()
    for i in range (0, len(scenes)-1):
        ds = segment_analysis(cds, int(scenes[i]), int(scenes[i+1]), metric=metric)
        ds_mean = np.mean(ds.samples)
        scene_correlations["scene_{0}".format(i+1)] = ds_mean
        scene_ds["scene_{0}".format(i+1)] = ds
        print("Finished scene {0}: Mean correlation was {1}".format(i, ds_mean))
        
    return scene_ds, scene_correlations




# Smooths the sums by doing a sliding window average.  
# TODO: detect scene boundaries. 
def detect_scenes(ds, window=5, a=0.01, n=34):
    
    X = ds.samples
    total_voxels = ds.shape[1]
    
    min_t = scipy.stats.t.ppf(1-a, n)    
    U = np.ma.masked_greater(X, min_t).mask  
    
    sums = np.array([np.sum(x) / float(total_voxels) for x in U])
    avgs = np.zeros_like(sums)

    plt.plot(sums)
    for i in range(window):
        buff = sums[:(i+1)]
        add = np.hstack((buff, sums[:-(i+1)]))
        avgs = avgs + add
    
    smoothed = avgs / float(window)
    fd_threshold = 0.10
    vertical_lines = []
    colors = []
    
    for i in range(len(smoothed)-1):
        x = smoothed[i]
        y = smoothed[i+1]
        if (y < fd_threshold and x > fd_threshold):            
            vertical_lines.append(i+1)
            colors.append('red')
        elif (x<fd_threshold and y>fd_threshold):
            vertical_lines.append(i-1)
            colors.append('blue')
    
    vertical_lines = (np.array(vertical_lines) * 2.5) / 60.0
    X = np.arange(len(smoothed)) * 2.5 / 60.0
    
    plt.clf()    
    plt.figure(figsize=(12,4))  
    plt.plot(X,smoothed)
    for i, line in enumerate(vertical_lines):
        plt.axvline(line, color=colors[i])
    plt.xlabel('Time (minutes)')
    plt.ylabel("% of brain 'activated' with a={0}".format(a))
    plt.show()


'''
    
    plot_title     Should be a string with {0} in it for the run number of each plot

'''    
def plot_activation_with_scenes(ds, scenes, plot_title, window=5, a=0.01, n=34):
    
    X = ds.samples
    total_voxels = ds.shape[1]
    
    min_t = scipy.stats.t.ppf(1-a, n)    
    U = np.ma.masked_greater(X, min_t).mask  
    
    sums = np.array([np.sum(x) / float(total_voxels) for x in U])
    avgs = np.zeros_like(sums)

    plt.plot(sums)
    for i in range(0,window):
        buff = sums[:(i+1)]
        add = np.hstack((buff, sums[:-(i+1)]))
        avgs = avgs + add
    
    smoothed = avgs / float(window)

    for run in range(1,4):    
        plt.clf()    
        fig = plt.figure(figsize=(12,4))  
        run_beg = (run-1)*211        
        run_end = run*211
        plt.plot(smoothed[run_beg:run_end])
        for i, line in enumerate(scenes):
            if (line < run_end and line > run_beg):            
                plt.axvline((line - run_beg), color='r')
        plt.xlabel('Time (2.5second slices)')
        plt.ylabel("% of brain 'activated' with a={0}".format(a))
        plt.title("Run {0} Brain Activation Compared to Scene Change".format(run))
        plt.show()
        fig.savefig(plot_title.format(run))

def find_common_activation_zones_at_scene_change(cds, scenes, padding=2):

    differences = dict()
    
    for i in range(0, len(scenes)-1):
        scene_change = scenes[i]
        voxels_before = np.mean(cds.samples[:,:,scene_change-padding:scene_change], axis=2)
        voxels_after = np.mean(cds.samples[:,:,scene_change:scene_change+padding], axis=2)
        
        t_values_before = ttest(voxels_before, popmean=0, alternative='greater')[0]
        t_values_after = ttest(voxels_after, popmean=0, alternative='greater')[0]
        
        samples = abs(t_values_before - t_values_after)
        samples = (samples - np.mean(samples)) / np.std(samples)
        ds = Dataset(samples)       
        ds.a = cds.a
        differences["scene_{0}".format(i+1)] = ds
        
    return differences
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

