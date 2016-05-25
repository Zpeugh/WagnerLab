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
    
    dslist = mult.get_2010_preprocessed_data(num_subjects=subjects, mask_path=mask_path, num_threads=n_cpu)
    cds = du.combine_datasets(dslist)
    
    ################################Radius 3#########################################
    cca_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='cca')
    
    f = open('results/data/cca_{0}_{1}.pckl'.format(subjects, radius), 'wb')
    pickle.dump(cca_res, f)
    f.close()
    
    du.export_to_nifti(cca_res, '{0}_cancorr_{1}_subject_r{2}'.format(brain_region,subjects, radius))
    
    
    corr_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='correlation')
    
    f = open('results/data/corr_{0}_{1}.pckl'.format(subjects, radius), 'wb')
    pickle.dump(corr_res, f)
    f.close()
    
    du.export_to_nifti(corr_res, '{0}_pearson_{1}_subject_r{2}'.format(brain_region,subjects, radius))
    
    
    du.plot_isc_vs_isi(corr_res, cca_res, '{0} {1} Subject ISI vs ISC: Searchlight Radius {2}'.format(brain_region, subjects, radius), save=True, filename='results/figures/{0}_{1}_{2}'.format(brain_region, subjects, radius))
