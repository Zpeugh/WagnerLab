# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:09:46 2016

@author: peugh.14
"""
import numpy as np
import matplotlib.pyplot as plt
from mvpa2.tutorial_suite import *
import fmri_preprocessing as fp


# This is the single function to call to get all of the preprocessed data.
def get_2010_preprocessed_data():
    
    
    INCORRECT_SR = 2.512    #The incorrect sample rate for 9 runs (samples/sec)
    CORRECT_SR = 2.5        #The proper sampling rate in samples/sec
   
    mask_path = 'masks/bigmask_3x3x3.nii'
    base_path = '/lab/neurodata/ddw/dartmouth/2010_SP/SUBJECTS/'
    run_path = "/FUNCTIONAL/swuabold{0}.nii"
    param_path = "/FUNCTIONAL/rp_abold{0}.txt"    
    
    
    ##The list of subject file names with the proper sampling rate of 2.5 s/sec
    subjects = ['0ctr_14oct09ft']#, '0ctr_30sep09kp']#, '0smk_17apr09ag', '0ctr_30sep09ef'] 
    #'0ctr_14oct09gl', '0ctr_30sep09sh', '0smk_22apr09cc','0ctr_14oct09js', '0ctr_30sep09so', 
    #'0ctr_18apr09yg', '0ctr_30sep09zl', '0ctr_19apr09tj', '0ctr_26jul09bc', '0smk_28sep09cb',
    #'0ctr_28sep09kr', '0ctr_28sep09sb', '0ctr_28sep09sg', '0ctr_29sep09ef', '0ctr_29sep09gp', 
    #'0ctr_29sep09mb', '0smk_13oct09ad', '0smk_31jul07sc_36slices', '0smk_07aug07lr_36slices',
    # '0smk_07jun07nw_36slices'  ]
    
    ##The list of subject file names with the improper sampling rate of 2.512 s/sec
    subjects_to_resample = ["0smk_02apr08jb", "0smk_06may08md"]#, "0smk_08may08kw"] 
    #"0smk_12may08ne", "0smk_12may08sb", "0smk_14mar07jm" "0smk_25feb08rl", 
    #"0smk_25feb08rz", "0smk_27feb08mi", "0smk_30may08sm" 
    
    ##The dictionary to contain the 34 subjects with 3 runs each
    dataset_dict = dict()
    
    ##store all of the normal datasets in the dictionary
    for index, subject in enumerate(subjects):
        
        # Load in the datasets for each run, adding the mask provided.
        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )    
        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path ) 
        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path ) 
      
        # Load in the parameters for each run
        params1 = np.loadtxt( base_path + subject + param_path.format(1) )
        params2 = np.loadtxt( base_path + subject + param_path.format(2) )
        params3 = np.loadtxt( base_path + subject + param_path.format(3) )    
       
        dm = fp.get_design_matrix([params1, params2, params3], 1)
        ds = fp.combineRuns([ds1, ds2, ds3],CORRECT_SR)     
        
        detrended_ds = fp.detrend_data_with_design_matrix(ds, dm)        
                    
        ds = fp.splice_ds_runs(detrended_ds,3,38,39)
        #zscore(ds1)
        dataset_dict[ "subject_{0}".format(index) ] = ds
        
        
    offset = len(subjects)
    
    #Resample and store all of the subjects who need resampling and place in the dataset_dict
#    for index, subject in enumerate(subjects_to_resample):  
#        
#        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )
#        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path )
#        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path )
#      
#        ds1 = fp.ds_resample( ds1, INCORRECT_SR, CORRECT_SR )
#        ds2 = fp.ds_resample( ds2, INCORRECT_SR, CORRECT_SR )
#        ds3 = fp.ds_resample( ds3, INCORRECT_SR, CORRECT_SR )    
#        
#        # Load in the parameters for each run
#        params1 = np.loadtxt( base_path + subject + param_path.format(1) )
#        params2 = np.loadtxt( base_path + subject + param_path.format(2) )
#        params3 = np.loadtxt( base_path + subject + param_path.format(3) )    
#       
#        detrended1 = fp.detrend_ds(ds1, params1)
#        detrended2 = fp.detrend_ds(ds2, params2)
#        detrended3 = fp.detrend_ds(ds3, params3)
#      
#        ds1 = fp.combineRuns([detrended1,detrended2,detrended3],CORRECT_SR,38, 39)
#        zscore(ds1)
#        dataset_dict[ "subject_{0}".format(index + offset) ] = ds1
        
    return dataset_dict
 
 


def voxel_plot(ds, voxel_position):
    ts = np.transpose(ds.samples)
    
    plt.figure(figsize=(10,6))
    plt.plot(ts[voxel_position])
#    plt.title("Timeseries for voxel {0}".format(ds.fa.voxel_indices[voxel_position]))    
    plt.title("Timeseries for voxel {0}".format(voxel_position))
    plt.axvline(211, color='r', linestyle='--')
    plt.axvline(422, color='r', linestyle='--')
    plt.show















