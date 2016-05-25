# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:46:17 2016

@author: peugh.14
"""


import multithreaded as mult
import dataset_utilities as du
import pickle

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
    dslist = mult.get_2010_preprocessed_data(num_subjects=subjects, mask_path=mask_path, num_threads=n_cpu)
    cds = du.combine_datasets(dslist)
    
    # Run CCA and save results to pickle and nifti file
    cca_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='cca')
    f = open('results/data/{0}_cca_{1}_{2}.pckl'.format(brain_region, subjects, radius), 'wb')
    pickle.dump(cca_res, f)
    f.close()   
    du.export_to_nifti(cca_res, 'results/nifti/{0}_cca_{1}s_r{2}'.format(brain_region,subjects, radius))

    # Run Pearson's correlation and save results to pickle and nifti file    
    corr_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='correlation')
    f = open('results/data/{0}_corr_{0}_{1}.pckl'.format(brain_region, subjects, radius), 'wb')
    pickle.dump(corr_res, f)
    f.close()
    du.export_to_nifti(corr_res, 'results/nifti/{0}_corr_{1}s_r{2}'.format(brain_region,subjects, radius))
    
    # Plot the colored and ISC vs ISI plot
    du.plot_colored_isc_vs_isi(corr_res, cca_res, voxels=corr_res.fa.voxel_indices, title='{0} {1} Subject ISI vs ISC: Searchlight Radius {2}'.format(brain_region, subjects, radius), save=True, filename='results/figures/{0}_{1}_{2}'.format(brain_region, subjects, radius))
