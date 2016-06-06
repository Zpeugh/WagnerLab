# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:46:17 2016

@author: peugh.14
"""


import multithreaded as mult
import dataset_utilities as du
import pickle
import matplotlib.pyplot as plt
import time
import timeit

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
    cds = mult.get_2010_preprocessed_data(num_subjects=subjects, mask_path=mask_path, num_threads=n_cpu)
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
    
    cds = mult.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
      
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

    cds = mult.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
    res = du.run_searchlight(cds, metric='pvalues', radius=radius, n_cpu=n_cpu)
    
    f = open('results/data/full_brain_p_values_r3.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    
    return res



def tvalues(num_subjects=34, radius=3, mask_path='masks/bigmask_3x3x3.nii', n_cpu=50):

    cds = mult.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
    res = du.run_searchlight(cds, metric='tvalues', radius=radius, n_cpu=n_cpu)
    
    f = open('results/data/full_brain_t_values_r2.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    
    return res
    

def timing_test(metric='cca', n_cpu=20, filename=None):

    cds = mult.get_2010_preprocessed_data(mask_path='masks/aal_l_ifg_oto_3x3x3.nii')
    
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
    
    
    
    
    
    
    
    
    
    
    
## Making masks
## open fslview
## save a mask
## flirt -in ../LeftFrontalMedialCortex.nii.gz -ref aal_l_amyg_3x3x3.nii -out left_medial_frontal -applyxfm -usesqform
## fslchfiletype NIFTI left_medial_frontal.nii.gz

