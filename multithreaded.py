# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:03:29 2016

@author: peugh.14
"""
import numpy as np
import matplotlib.pyplot as plt
from mvpa2.tutorial_suite import *
import fmri_preprocessing as fp
import time
from multiprocessing import Pool


INCORRECT_SR = 2.5112    #The incorrect sample rate for 9 runs (samples/sec)
CORRECT_SR = 2.5        #The proper sampling rate in samples/sec

#MASK_PATH = 'masks/aal_l_hippocampus_3x3x3.nii'
BASE_PATH = '/lab/neurodata/ddw/dartmouth/2010_SP/SUBJECTS/'
RUN_PATH = "/FUNCTIONAL/swuabold{0}.nii"
PARAM_PATH = "/FUNCTIONAL/rp_abold{0}.txt"    

##The list of subject file names with the proper sampling rate of 2.5 s/sec
subjects = ['0ctr_14oct09ft', '0ctr_30sep09kp', '0smk_17apr09ag', '0ctr_30sep09ef', 
'0ctr_14oct09gl', '0ctr_30sep09sh', '0smk_22apr09cc','0ctr_14oct09js', '0ctr_30sep09so', 
'0ctr_18apr09yg', '0ctr_30sep09zl', '0ctr_19apr09tj', '0ctr_26jul09bc', '0smk_28sep09cb',
'0ctr_28sep09kr', '0ctr_28sep09sb', '0ctr_28sep09sg', '0ctr_29sep09ef', '0ctr_29sep09gp', 
'0ctr_29sep09mb', '0smk_13oct09ad', '0smk_31jul07sc_36slices', '0smk_07aug07lr_36slices',
'0smk_07jun07nw_36slices', "0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw", 
"0smk_12may08ne", "0smk_12may08sb", "0smk_14mar07jm", "0smk_25feb08rl", 
"0smk_25feb08rz", "0smk_27feb08mi", "0smk_30may08sm" ]


  
##The dictionary to contain the 34 subjects with 3 runs each

offset = len(subjects) #number of subjects that don't need resampled


def subject_needs_resampled(subject):
    
    return subject in ["0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw", "0smk_12may08ne", 
            "0smk_12may08sb", "0smk_14mar07jm", "0smk_25feb08rl", "0smk_25feb08rz", 
            "0smk_27feb08mi", "0smk_30may08sm" ]
    

def get_ds(subject,index, mask_path, degrees):        
         
        ds1 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(1), mask=mask_path )
        ds2 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(2), mask=mask_path )
        ds3 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(3), mask=mask_path )
      
          
        if subject_needs_resampled(subject):            
            ds1 = fp.ds_resample( ds1, INCORRECT_SR, CORRECT_SR )
            ds2 = fp.ds_resample( ds2, INCORRECT_SR, CORRECT_SR )            
            ds3 = fp.ds_resample( ds3, INCORRECT_SR, CORRECT_SR )    
        
        params1 = np.loadtxt( BASE_PATH + subject + PARAM_PATH.format(1) )
        params2 = np.loadtxt( BASE_PATH + subject + PARAM_PATH.format(2) )
        params3 = np.loadtxt( BASE_PATH + subject + PARAM_PATH.format(3) )    
      
        dm = fp.get_design_matrix([params1, params2, params3], 1)
        ds = fp.combineRuns([ds1, ds2, ds3],CORRECT_SR)
        
        ds = fp.detrend_data_with_design_matrix(ds, dm)                   
        ds = fp.splice_ds_runs(ds,3,38,39)
        zscore(ds, chunks_attr="chunks")
        
        ds.a = ds1.a

        return ds


def multiple_get_ds(arg_list):
    return get_ds(*arg_list)

    
    

'''====================================================================================
    Get the combined, resampled, sliced, detrended and normalized Datasets.  
    
    num_subjects         (optional) The number of samples 1-34 to work with.  
                        Default is all samples (34).
    mask_path           (optional) The path to the .nii mask file to use on all 
                        of the samples. Default is bigmask_3x3x3.nii
    degrees             (optional) The number of polynomial degrees to use when
                        detrending the dataset
                           
    Returns             the List of num_subj subjects' preprocessed datasets
======================================================================================'''
def get_2010_preprocessed_data(num_subjects=34, mask_path='masks/bigmask_3x3x3.nii', degrees=1, num_threads=34):
    
    args_list = []
    pool = Pool(num_threads)    
    
    t_0 = time.time()
    ##store all of the normal datasets in the dictionary
    for index, subject in enumerate(subjects):
 
        if index < num_subjects:
            args_list.append( (subject, index, mask_path, degrees,) )           
          
    results = pool.map(multiple_get_ds, args_list)   
        
    t_elapsed = time.time() - t_0
    print("Total time to preprocess: %.3f seconds" % (t_elapsed) ) 
    print("Number of subjects: %d" % num_subjects )
    print("Average time per subject: %.4f" % (t_elapsed / float(num_subjects) ))
    print("Mask used: %s" % mask_path )
    
    return results    
                  
       
 
 

 
 
 
 
 
 
 
 