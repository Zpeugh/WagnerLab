# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:03:29 2016

@author: peugh.14

@description: Module to quickly parallel preprocess the 2010 movie data

"""

import fmri_preprocessing as fp
import dataset_utilities as du
from mvpa2.tutorial_suite import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import time

    
    
INCORRECT_SR = 2.5112    #The incorrect sample rate for 10 runs (samples/sec)
CORRECT_SR = 2.5        #The proper sampling rate in samples/sec

BASE_PATH = '/lab/neurodata/ddw/dartmouth/2010_SP/SUBJECTS/'
RUN_PATH = "/FUNCTIONAL/swuabold{0}.nii"
PARAM_PATH = "/FUNCTIONAL/rp_abold{0}.txt"    

SUBJECTS = ['0ctr_14oct09ft', '0ctr_14oct09gl', '0ctr_14oct09js', '0ctr_18apr09yg','0ctr_19apr09tj', '0ctr_26jul09bc', '0ctr_28sep09kr', '0ctr_28sep09sb','0ctr_28sep09sg', '0ctr_29sep09ef', '0ctr_29sep09gp', 
'0ctr_29sep09mb', '0ctr_30sep09ef', '0ctr_30sep09kp', '0ctr_30sep09sh', '0ctr_30sep09so','0ctr_30sep09zl', '0smk_02apr08jb', '0smk_06may08md', '0smk_07aug07lr_36slices','0smk_07jun07nw_36slices', '0smk_08may08kw', 
'0smk_12may08ne', '0smk_12may08sb', '0smk_13oct09ad', '0smk_14mar07jm',
'0smk_17apr09ag', '0smk_22apr09cc', '0smk_25feb08rl', '0smk_25feb08rz',
'0smk_27feb08mi', '0smk_28sep09cb', '0smk_30may08sm', '0smk_31jul07sc_36slices']

#SUBJECTS = ['0smk_02apr08jb']


def _subject_needs_resampled(subject):
    
    return subject in ["0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw", "0smk_12may08ne", 
            "0smk_12may08sb", "0smk_14mar07jm", "0smk_25feb08rl", "0smk_25feb08rz", 
            "0smk_27feb08mi", "0smk_30may08sm" ]
    

def _get_ds(subject,index, mask_path, degrees):                
        
        ds1 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(1), mask=mask_path )
        ds2 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(2), mask=mask_path )
        ds3 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(3), mask=mask_path )
        
        print(ds1.shape)
        if _subject_needs_resampled(subject):            
            ds1 = fp.ds_resample( ds1, INCORRECT_SR, CORRECT_SR )
            ds2 = fp.ds_resample( ds2, INCORRECT_SR, CORRECT_SR )            
            ds3 = fp.ds_resample( ds3, INCORRECT_SR, CORRECT_SR )    
        
        params1 = np.loadtxt( BASE_PATH + subject + PARAM_PATH.format(1) )
        params2 = np.loadtxt( BASE_PATH + subject + PARAM_PATH.format(2) )
        params3 = np.loadtxt( BASE_PATH + subject + PARAM_PATH.format(3) )    
      
        dm = fp.get_design_matrix([params1, params2, params3], 1)
        ds = fp.combine_runs([ds1, ds2, ds3],CORRECT_SR)                

        ds = fp.detrend_data_with_design_matrix(ds, dm) 
        ds = fp.splice_ds_runs(ds,3,38,39)
                
        zscore(ds, chunks_attr="chunks")
        
        ds.a.mapper = ds1.a.mapper
     
        ds.sa["subject"] = np.zeros(ds.shape[0]) + index

        print("Finished subject: {0}".format(index))
        return ds        
        #result_list[index] = ds

# Simple wrapper function to call _get_ds multiple times
def _multiple_get_ds(arg_list):
    return _get_ds(*arg_list)

    

def new_get_2010_preprocessed_data(num_subjects=34, mask_path='../fmri/masks/bigmask_3x3x3.nii', degrees=1, num_threads=34,  combine=True, verbose=False):
    
    args_list = []
    processes = []
    results = []
    pool = Pool(num_threads)    
    
    t_0 = time.time()
    
    for index, subject in enumerate(SUBJECTS):
 
        if index < num_subjects:
            processes.append(pool.apply_async(_get_ds, [subject, index, mask_path, degrees]))     
    
    pool.close()
    pool.join()
    
    for proc in processes:
        results.append(proc.get())
        
    t_elapsed = time.time() - t_0
    
    if verbose:
        print("Total time to preprocess: %.3f seconds" % (t_elapsed) ) 
        print("Number of subjects: %d" % num_subjects )
        print("Average time per subject: %.4f" % (t_elapsed / float(num_subjects) ))
        print("Mask used: %s" % mask_path )
    
    
    if combine:      
        if verbose:
            print("combining datasets")
        t_0 = time.time()
        cds = du.combine_datasets(results)
        t_elapsed = time.time() - t_0
        if verbose:
            print("combining datasets took {0} seconds".format(t_elapsed))
        if num_subjects == 34:
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/like_34.txt') as f:
                cds.sa["likes"] = map(float, f)    
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/trans_34.txt') as f:
                cds.sa["transportation"] = map(float, f)  
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/transddw_34.txt') as f:
                cds.sa["transportation"] = map(float, f)  
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/traits_34.txt') as f:
                cds.sa["traits"] = map(float, f)  
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/acting_34.txt') as f:
                cds.sa["acting"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/actorlike_34.txt') as f:
                cds.sa["actor_like"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIec_34.txt') as f:
                cds.sa["IRIec"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIfs_34.txt') as f:
                cds.sa["IRIfs"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIpd_34.txt') as f:
                cds.sa["IRIpd"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIpt_34.txt') as f:
                cds.sa["IRIpt"] = map(float, f)
        return cds   
    else:
        return results        


def get_2010_scene_splits():
    SCENE_CHANGE_FILE = "/lab/neurodata/zachs_work/wagner_2010/scene_changes/scenes_run{0}.txt"
    SEC_PER_MIN = 60
    RUN_BEG_CUTOFF = 38
    SAMPLES_PER_RUN = 211
    RUN_END_CUTOFF = 288-39
    
    scene_change_times = []
    
    for run in range(1,4):
        offset = (run-1) * SAMPLES_PER_RUN
        with open(SCENE_CHANGE_FILE.format(run), 'r') as f:
            for line in f:
                line = line.strip()
                times = line.split(':')
                
                
                minutes = int(times[0])
                seconds = float(times[1])
                    
                total_seconds = minutes * SEC_PER_MIN + seconds
                total_time = total_seconds / float(2.5)
                
                if (total_time > RUN_BEG_CUTOFF and total_time < RUN_END_CUTOFF):
                    scene_change_times.append(total_time + offset - RUN_BEG_CUTOFF)
    return scene_change_times
            
                
            
            
        
'''====================================================================================
    Get the combined, resampled, sliced, detrended and normalized Datasets, stacked and
    put together into one Dataset
    
    num_subjects         (optional) The number of samples 1-34 to work with.  
                        Default is all samples (34).
    mask_path           (optional) The path to the .nii mask file to use on all 
                        of the samples. Default is bigmask_3x3x3.nii
    degrees             (optional) The number of polynomial degrees to use when
                        detrending the dataset
    num_threads         The number of different threads to create, will default to 34
                        which assumes all subjects are being run in parellel.                     
    combine             (optional) Whether or not to return a single Dataset of all
                        subjects combined (True, default) or a list containing 
                        each subject (False).
    verbose             (optional) True to print some information to the screen
                           
    Returns             the Dataset of num_subj subjects' preprocessed datasets. This
                        has shape (#subjects, #voxels_in_mask, #time_samples).
======================================================================================'''
def get_2010_preprocessed_data(num_subjects=34, mask_path='../fmri/masks/bigmask_3x3x3.nii', degrees=1, num_threads=34,  combine=True, verbose=False):
    
    args_list = []
    pool = Pool(num_threads)    
    
    t_0 = time.time()
    
    for index, subject in enumerate(SUBJECTS):
 
        if index < num_subjects:
            args_list.append( (subject, index, mask_path, degrees,) )           
      
    results = pool.map(_multiple_get_ds, args_list)   
        
    t_elapsed = time.time() - t_0
    
    if verbose:
        print("Total time to preprocess: %.3f seconds" % (t_elapsed) ) 
        print("Number of subjects: %d" % num_subjects )
        print("Average time per subject: %.4f" % (t_elapsed / float(num_subjects) ))
        print("Mask used: %s" % mask_path )
    
    
    if combine:      
        if verbose:
            print("combining datasets")
        t_0 = time.time()
        cds = du.combine_datasets(results)
        t_elapsed = time.time() - t_0
        if verbose:
            print("combining datasets took {0} seconds".format(t_elapsed))
        if num_subjects == 34:
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/like_34.txt') as f:
                cds.sa["likes"] = map(float, f)    
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/trans_34.txt') as f:
                cds.sa["transportation"] = map(float, f)  
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/transddw_34.txt') as f:
                cds.sa["transportation"] = map(float, f)  
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/traits_34.txt') as f:
                cds.sa["traits"] = map(float, f)  
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/acting_34.txt') as f:
                cds.sa["acting"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/actorlike_34.txt') as f:
                cds.sa["actor_like"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIec_34.txt') as f:
                cds.sa["IRIec"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIfs_34.txt') as f:
                cds.sa["IRIfs"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIpd_34.txt') as f:
                cds.sa["IRIpd"] = map(float, f)
            with open('/lab/neurodata/ddw/dartmouth/2010_SP/ROI/VARIABLES/IRIpt_34.txt') as f:
                cds.sa["IRIpt"] = map(float, f)
        return cds   
    else:
        return results              
       


'''====================================================================================
    Get the combined, resampled Datasets that are not detrended.  Currently does not
    run in parallel with multiple threads so it can be very slow. 
    
    num_samples         (optional) The number of samples 1-34 to work with.  
                        Default is all samples (34).
    mask_path           (optional) The path to the .nii mask file to use on all 
                        of the samples. Desfault is bigmask_3x3x3.nii
    slice_samples       (optional) Tells whether or not to slice out the begginning 
                        and ending samples. Default is true
                        
    Returns             the dictionary of num_subj subjects.  Each subject's dataset 
                        can be accessed with ["subject_0"], ["subject_1], etc.
======================================================================================'''
def get_raw_2010_datasets(num_samples=34, mask_path='../fmri/masks/bigmask_3x3x3.nii', slice_samples=True):
    
    incorrect_sr = 2.5112    #The incorrect sample rate for 9 runs (samples/sec)
    correct_sr = 2.5        #The proper sampling rate in samples/sec
   
    mask_path = '../fmri/masks/bigmask_3x3x3.nii'
    base_path = '/lab/neurodata/ddw/dartmouth/2010_SP/SUBJECTS/'
    run_path = "/FUNCTIONAL/swuabold{0}.nii"
    param_path = "/FUNCTIONAL/rp_abold{0}.txt"    
    
    
    ##The list of subject file names with the proper sampling rate of 2.5 s/sec
    subjects = ['0ctr_14oct09ft', '0ctr_30sep09kp', '0smk_17apr09ag', '0ctr_30sep09ef', 
    '0ctr_14oct09gl', '0ctr_30sep09sh', '0smk_22apr09cc','0ctr_14oct09js', '0ctr_30sep09so', 
    '0ctr_18apr09yg', '0ctr_30sep09zl', '0ctr_19apr09tj', '0ctr_26jul09bc', '0smk_28sep09cb',
    '0ctr_28sep09kr', '0ctr_28sep09sb', '0ctr_28sep09sg', '0ctr_29sep09ef', '0ctr_29sep09gp', 
    '0ctr_29sep09mb', '0smk_13oct09ad', '0smk_31jul07sc_36slices', '0smk_07aug07lr_36slices',
    '0smk_07jun07nw_36slices' ]
    
    ##The list of subject file names with the improper sampling rate of 2.512 s/sec
    subjects_to_resample = ["0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw", 
    "0smk_12may08ne", "0smk_12may08sb", "0smk_14mar07jm" "0smk_25feb08rl", 
    "0smk_25feb08rz", "0smk_27feb08mi", "0smk_30may08sm" ]
    
    ##The dictionary to contain the 34 subjects with 3 runs each
    dataset_dict = dict()
    offset = len(subjects) #number of subjects that don't need resampling
    
    ##store all of the normal datasets in the dictionary
    for index, subject in enumerate(subjects):
        print("Preprocessing subject {0}".format(index))        
             
        if index >= num_samples:
            return dataset_dict
                   
        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )    
        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path ) 
        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path ) 
      
        ds = fp.combine_runs([ds1, ds2, ds3],correct_sr)
        
        ds.a = ds1.a
        dataset_dict[ "subject_{0}".format(index) ] = fp.splice_ds_runs(ds,3,38,39)
    
    #Resample and store all of the subjects who need resampling and place in the dataset_dict
    for index, subject in enumerate(subjects_to_resample): 
        print("Preprocessing subject {0}".format(index + offset))       
        if index + offset >= num_samples:
            return dataset_dict
            
        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )
        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path )
        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path )
      
        ds1 = fp.ds_resample( ds1, incorrect_sr, correct_sr )
        ds2 = fp.ds_resample( ds2, incorrect_sr, correct_sr )            
        ds3 = fp.ds_resample( ds3, incorrect_sr, correct_sr )    
        
        ds = fp.combine_runs([ds1, ds2, ds3],correct_sr)        
        
        ds.a.mapper = ds1.a.mapper
        dataset_dict["subject_{0}".format(index + offset)] = fp.splice_ds_runs(ds,3,38,39)
        
    return dataset_dict

 
 
 
 
 
 
 