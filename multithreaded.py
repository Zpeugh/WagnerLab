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


incorrect_sr = 2.5112    #The incorrect sample rate for 9 runs (samples/sec)
correct_sr = 2.5        #The proper sampling rate in samples/sec
   
mask_path = 'masks/bigmask_3x3x3.nii'
#mask_path = 'masks/aal_l_hippocampus_3x3x3.nii'
base_path = '/lab/neurodata/ddw/dartmouth/2010_SP/SUBJECTS/'
run_path = "/FUNCTIONAL/swuabold{0}.nii"
param_path = "/FUNCTIONAL/rp_abold{0}.txt"    

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
dataset_dict = dict()
offset = len(subjects) #number of subjects that don't need resampled


def subject_needs_resampled(subject):
    subs = ["0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw", "0smk_12may08ne", 
            "0smk_12may08sb", "0smk_14mar07jm", "0smk_25feb08rl", "0smk_25feb08rz", 
            "0smk_27feb08mi", "0smk_30may08sm" ]
            
    if subject in subs:
        return True
    else:
        return False        
  

def get_ds(subject,index, degrees):        
         
        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )
        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path )
        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path )
      
          
        if subject_needs_resampled(subject):            
            ds1 = fp.ds_resample( ds1, incorrect_sr, correct_sr )
            ds2 = fp.ds_resample( ds2, incorrect_sr, correct_sr )            
            ds3 = fp.ds_resample( ds3, incorrect_sr, correct_sr )    
        
        params1 = np.loadtxt( base_path + subject + param_path.format(1) )
        params2 = np.loadtxt( base_path + subject + param_path.format(2) )
        params3 = np.loadtxt( base_path + subject + param_path.format(3) )    
      
        dm = fp.get_design_matrix([params1, params2, params3], 1)
        ds = fp.combineRuns([ds1, ds2, ds3],correct_sr)
        
        ds = fp.detrend_data_with_design_matrix(ds, dm)                   
        ds = fp.splice_ds_runs(ds,3,38,39)
        zscore(ds, chunks_attr="chunks")
        
        ds.a = ds1.a

        dataset_dict[ "dm_{0}".format(index) ] = dm
        dataset_dict[ "subject_{0}".format(index) ] = ds
        return ds


def multiple_get_ds(arg_list):
    return get_ds(*arg_list)

    
    

'''====================================================================================
    Get the combined, resampled, sliced, detrended and normalized Datasets.  
    
    num_samples         (optional) The number of samples 1-34 to work with.  
                        Default is all samples (34).
    mask_path           (optional) The path to the .nii mask file to use on all 
                        of the samples. Default is bigmask_3x3x3.nii
    degrees             (optional) The number of polynomial degrees to use when
                        detrending the dataset
                           
    Returns             the dictionary of num_subj subjects.  Each subject's dataset 
                        can be accessed with ["subject_0"], ["subject_1], etc.
                        Additionally, the design matrices of the subjects can be
                        accessed via ["dm_0"], ["dm_1"], etc.
======================================================================================'''
def get_2010_preprocessed_data(num_samples=34, mask_path='masks/bigmask_3x3x3.nii', degrees=1, num_threads = 34):
    
    args_list = []
    pool = Pool(num_threads)    
    
    t_0 = time.clock()
    ##store all of the normal datasets in the dictionary
    for index, subject in enumerate(subjects):
 
        if index <= num_samples:
            args_list.append( (subject, index, degrees,) )           
          
    results = pool.map(multiple_get_ds, args_list)   
        
        
    print("It took {0} seconds to preprocess all 34 subjects with bigmask.".format(time.clock() - t_0))    
    return results    
                  
       
 
 

 
 
 
 
 
 
 
 