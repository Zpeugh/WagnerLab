# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:46:17 2016

@author: peugh.14
"""


import load_data as ld
import dataset_utilities as du
import pickle
import matplotlib.pyplot as plt
import time
import timeit
from mvpa2.suite import Dataset
import numpy as np

'''=======================================================================================
    Run intersubject correlation (Pearson's r) and intersubject information 
    (canonical correlation analysis) on the Haxby_2010 dataset and save a pickle of the
    resulting Dataset for each, along with nifti files and the graph of ISC vs ISI.
    
    radius           the radius of the searchlight in voxels
    n_cpu            the number of cpu cores to run on
    subjects         number of subjects to test on
    brain_region     the string prefix of files to explain what region of the brain
                     is being tested.  i.e. 'left_hemisphere'
    mask_path        the path to the .nii mask for the corresponding brain region
=======================================================================================''' 
def run_cca_and_isc(radius, n_cpu, subjects, brain_region, mask_path):
    
    # Get the datasets
    cds = ld.get_2010_preprocessed_data(num_subjects=subjects, mask_path=mask_path, num_threads=n_cpu)
    # Run CCA and save results to pickle and nifti file
    cca_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='cca')
    f = open('results/data/{0}_cca_{1}_{2}.pckl'.format(brain_region, subjects, radius), 'wb')
    pickle.dump(cca_res, f)
    f.close()   
    du.export_to_nifti(cca_res, 'results/nifti/{0}_cca_{1}s_r{2}'.format(brain_region,subjects, radius))

    # Run Pearson's correlation and save results to pickle and nifti file    
    corr_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='correlation')
    f = open('results/data/{0}_corr_{1}_{2}.pckl'.format(brain_region, subjects, radius), 'wb')
    pickle.dump(corr_res, f)
    f.close()
    du.export_to_nifti(corr_res, 'results/nifti/{0}_corr_{1}s_r{2}'.format(brain_region,subjects, radius))
    
    # Plot the colored and ISC vs ISI plot
    du.plot_colored_isc_vs_isi(corr_res, cca_res, voxels=corr_res.fa.voxel_indices, title='{0} {1} Subject ISI vs ISC: Searchlight Radius {2}'.format(brain_region, subjects, radius), save=True, filename='results/figures/{0}_{1}_{2}'.format(brain_region, subjects, radius))    




def validation_bargraph(num_subjects, mask_path, radii=[0,1,2,3,4,5], n_cpu=None):
    
    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
      
    cancorrs = []
    max_corrs = []
    for rad in radii:
        t_0 = time.time()
        res = du.run_searchlight(cds, metric='cca_validate', radius=rad, n_cpu=n_cpu)
        means = res.samples.mean(axis=1)
        cancorrs.append(means[0])
        max_corrs.append(means[1])
        t_elapsed = time.time() - t_0
        print("Done with radius {0}\nTook {1} seconds".format(rad, t_elapsed))

    plt.clf()
    
    plt.bar(radii, cancorrs, color='steelblue')
    plt.xlabel('Searchlight radius (voxels)')
    plt.ylabel('Average First Canonical Correlation')
    plt.title('First Canonical Correlation as Searchlight Radius Increases')
    plt.show()
    
    
    plt.bar(radii, max_corrs, color='orangered')
    plt.xlabel('Searchlight radius (voxels)')
    plt.ylabel('Average Maximum Prediction Correlation')
    plt.title('Cross Validation and Prediction Average Maximum Accuracy')
    plt.show()
    
    return (cancorrs, max_corrs)
    
    
    

def pvalues(num_subjects=34, radius=3, mask_path='masks/bigmask_3x3x3.nii', n_cpu=50):

    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
    res = du.run_searchlight(cds, metric='pvalues', radius=radius, n_cpu=n_cpu)
    
    f = open('results/data/full_brain_p_values_r3.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    
    return res



def tvalues(num_subjects=34, radius=3, mask_path='masks/bigmask_3x3x3.nii', n_cpu=50):

    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
    res = du.run_searchlight(cds, metric='tvalues', radius=radius, n_cpu=n_cpu)
    
    f = open('results/data/full_brain_t_values_r2.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    
    return res
    

def timing_test(metric='cca', n_cpu=20, filename=None):

    cds = ld.get_2010_preprocessed_data(mask_path='masks/aal_l_ifg_oto_3x3x3.nii')
    
    times = []     
    for rad in range(7):
        t_0 = time.time()        
        du.run_searchlight(cds, radius=rad, n_cpu=n_cpu)
        times.append((time.time()-t_0) * n_cpu)
    [1,5.3,20,53,86]
    plt.clf()
    fig = plt.figure(figsize=(10,6))
    plt.plot(times, '-o')
    plt.xlabel('Searchlight Radius (voxels)')
    plt.ylabel('Single Core Time (seconds)')
    plt.title('CCA Time as a Function of Searchlight Radius')
    if filename:
       fig.savefig(filename) 
    plt.show()
    
    return times
    
def compare_transportation_groups(cds, high=5.5, low=3, metric='correlation', radius=2, n_cpu=20):
    
    num_samples = cds.shape[2]
    num_voxels = cds.shape[1]
    
    high_indices = [i for i, x in enumerate(cds.sa.transportation) if x>=high]
    low_indices = [i for i, x in enumerate(cds.sa.transportation) if x<=low]
    
    print("\n{0} subjects have scores above {1}".format(len(high_indices), high))
    print("{0} subjects have scores below {1}".format(len(low_indices), low))    
    
    # Get the Dataset for subjects with high 'transportation' scores
    high_tup = ()
    for i in high_indices:
        high_tup = high_tup + (cds.samples[i, :, :],)
    
    high_ds = Dataset(np.vstack(high_tup).reshape((len(high_indices), num_voxels, num_samples)))    
    high_ds.a.mapper = cds.a.mapper
    high_ds.fa["voxel_indices"] = cds.fa.voxel_indices    
   
    # Get the Dataset for subjects with low 'transportation' scores
    low_tup = ()
    for i in low_indices:
        low_tup = low_tup + (cds.samples[i, :, :],)
    
    low_ds = Dataset(np.vstack(low_tup).reshape((len(low_indices), num_voxels, num_samples)))    
    low_ds.a.mapper = cds.a.mapper
    low_ds.fa["voxel_indices"] = cds.fa.voxel_indices


    high_res = du.run_searchlight(high_ds, radius=radius, n_cpu=n_cpu, metric=metric)
    low_res = du.run_searchlight(low_ds, radius=radius, n_cpu=n_cpu, metric=metric)
    
    
    print("High transportation average {0} value: {1}".format(metric, high_res.samples.mean()))    
    print("Low transportation average {0} value: {1}".format(metric, low_res.samples.mean()))   
    
    return high_res, low_res
    


    
def test_all_pairs(cds, radius=2, n_cpu=20, metric='correlation', method='multiply', feature='transportation'):
    
   
    X = du.pairwise_feature_list(cds, feature=feature, method=method)
    
    if metric == 'correlation':
        res = du.run_searchlight(cds, radius=radius, n_cpu=n_cpu, metric='all_pearsons')
    elif metric == 'cca':
        res = du.run_searchlight(cds, radius=radius, n_cpu=n_cpu, metric='all_cca')
    else:
        print("INVALID METRIC.  Choose 'correlation' or 'cca'")
    means = np.mean(res.samples, axis=1)   
    
    plt.plot(X, means, 'x')
    
    return res
    
    
def plot_feature_vs_result(cds, means, feature, title, method='multiply'):

    X = du.pairwise_feature_list(cds, feature, method=method)
    
    plt.plot(X, means, 'o')
    plt.xlabel(feature)   
    plt.ylabel('Correlation')
    plt.title(title)
    plt.show
    
    print("Correlation: {0}".format(np.corrcoef(X, means)[0,1]))



    
## Making masks
## open fslview
## save a mask
## flirt -in ../LeftFrontalMedialCortex.nii.gz -ref aal_l_amyg_3x3x3.nii -out left_medial_frontal -applyxfm -usesqform
## fslchfiletype NIFTI left_medial_frontal.nii.gz

