# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:09:46 2016

@author: peugh.14
"""
import numpy as np
import matplotlib.pyplot as plt
from mvpa2.tutorial_suite import *
import fmri_preprocessing as fp
from scipy.stats import pearsonr
from mvpa2.measures.searchlight import sphere_searchlight


'''====================================================================================
    Get the combined, resampled, sliced, detrended and normalized Datasets.  
    
    num_samples         (optional) The number of samples 1-34 to work with.  
                        Default is all samples (34).
    mask_path           (optional) The path to the .nii mask file to use on all 
                        of the samples. Desfault is bigmask_3x3x3.nii
    degrees             (optional) The number of polynomial degrees to use when
                        detrending the dataset
                           
    Returns             the dictionary of num_subj subjects.  Each subject's dataset 
                        can be accessed with ["subject_0"], ["subject_1], etc.
                        Additionally, the design matrices of the subjects can be
                        accessed via ["dm_0"], ["dm_1"], etc.
======================================================================================'''
def get_2010_preprocessed_data(num_samples=34, mask_path='masks/bigmask_3x3x3.nii', degrees=1):
    
    
    incorrect_sr = 2.512    #The incorrect sample rate for 9 runs (samples/sec)
    correct_sr = 2.5        #The proper sampling rate in samples/sec
   
    #mask_path = 'masks/bigmask_3x3x3.nii'
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
    '0smk_07jun07nw_36slices' ]
    
    ##The list of subject file names with the improper sampling rate of 2.512 s/sec
    subjects_to_resample = ["0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw", 
    "0smk_12may08ne", "0smk_12may08sb", "0smk_14mar07jm" "0smk_25feb08rl", 
    "0smk_25feb08rz", "0smk_27feb08mi", "0smk_30may08sm" ]
    
    ##The dictionary to contain the 34 subjects with 3 runs each
    dataset_dict = dict()
    offset = len(subjects) #number of subjects that don't need resampled
    
    ##store all of the normal datasets in the dictionary
    for index, subject in enumerate(subjects):
        
        t_0 = time.clock()          
        print("Preprocessing subject {0}...".format(index))        
        if index >= num_samples:
            return dataset_dict
        # Load in the datasets for each run, adding the mask provided.
        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )    
        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path ) 
        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path ) 
      
        # Load in the parameters for each run
        params1 = np.loadtxt( base_path + subject + param_path.format(1) )
        params2 = np.loadtxt( base_path + subject + param_path.format(2) )
        params3 = np.loadtxt( base_path + subject + param_path.format(3) )    
      
        dm = fp.get_design_matrix([params1, params2, params3], degrees)       
        ds = fp.combineRuns([ds1, ds2, ds3],correct_sr)     
        
        ds = fp.detrend_data_with_design_matrix(ds, dm)                   
        ds = fp.splice_ds_runs(ds,3,38,39)
        zscore(ds, chunks_attr="chunks")
        
        ds.a = ds1.a
        dataset_dict[ "dm_{0}".format(index) ] = dm
        dataset_dict[ "subject_{0}".format(index) ] = ds
        print (time.clock() - t_0)

    
    
    #Resample and store all of the subjects who need resampling and place in the dataset_dict
    for index, subject in enumerate(subjects_to_resample):  
        t_0 = time.clock()             
        print("Preprocessing subject {0}...".format(index + offset))  
        if index + offset >= num_samples:
            return dataset_dict
            
        ds1 = fmri_dataset( base_path + subject + run_path.format(1), mask=mask_path )
        ds2 = fmri_dataset( base_path + subject + run_path.format(2), mask=mask_path )
        ds3 = fmri_dataset( base_path + subject + run_path.format(3), mask=mask_path )
      
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
        dataset_dict[ "subject_{0}".format(index + offset) ] = ds
        print (time.clock() - t_0)
        
    return dataset_dict
 
 

'''====================================================================================
    Get the combined, resampled Datasets that are not detrended.
    
    num_samples         (optional) The number of samples 1-34 to work with.  
                        Default is all samples (34).
    mask_path           (optional) The path to the .nii mask file to use on all 
                        of the samples. Desfault is bigmask_3x3x3.nii
    slice_samples       (optional) Tells whether or not to slice out the begginning 
                        and ending samples. Default is true
                        
    Returns             the dictionary of num_subj subjects.  Each subject's dataset 
                        can be accessed with ["subject_0"], ["subject_1], etc.
======================================================================================'''
def get_raw_2010_datasets(num_samples=34, mask_path='masks/bigmask_3x3x3.nii', slice_samples=True):
    
    incorrect_sr = 2.512    #The incorrect sample rate for 9 runs (samples/sec)
    correct_sr = 2.5        #The proper sampling rate in samples/sec
   
    mask_path = 'masks/bigmask_3x3x3.nii'
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
      
        ds = fp.combineRuns([ds1, ds2, ds3],correct_sr)
        
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
        
        ds = fp.combineRuns([ds1, ds2, ds3],correct_sr)        
        
        ds.a.mapper = ds1.a.mapper
        dataset_dict["subject_{0}".format(index + offset)] = fp.splice_ds_runs(ds,3,38,39)
        
    return dataset_dict


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
    Plot the timeseries of a single voxel for a Dataset using matplotlib.
    
    ds                  The Dataset object containing samples
    voxel_position      a number representing which voxel in the dataset to display
======================================================================================'''    
def fourier_plot(ds, voxel_position):
    plt.clf()    
    plt.figure(figsize=(10,6))
    plt.plot(np.fft.fft(np.transpose(ds.samples)[voxel_position]))
    plt.title("Timeseries for voxel {0}".format(voxel_position))
    plt.axvline(211, color='r', linestyle='--')
    plt.axvline(422, color='r', linestyle='--')
    plt.show()

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
    Returns     a list of all of the Dataset objects in the dictionary
======================================================================================'''    
def ds_dict_to_list(dataset_dict):
    return [dataset_dict[key] for key in dataset_dict if key.find("subject") != -1 ]
    
 
   
'''====================================================================================
    The average absolute valued Pearson's correlation coefficient between two subjects
    
    subj1               The Dataset for subject 1.
                       
    subj2               The Dataset for subject 2.
                        
    Returns             The mean |r| value between all aubjects time series
======================================================================================'''
def average_pearsons_r(subj1, subj2):
    x = np.transpose(subj1.samples)
    y = np.transpose(subj2.samples)
    
    return np.mean( [ abs(pearsonr(x[i], y[i])[0]) for i in range(len(x))] )



'''====================================================================================
    Get the average Pearson's correlation coefficient between two subjects for all
    possible pair combinations in the list of Datasets.
    
    ds_list      The iterable list of Dataset objects containing subject data. 
                        
    Returns      The mean value of all of the pair combinations of Pearson's |r| values
======================================================================================'''
def pearsons_ISC(ds_list):
    
    dl = ds_list
    if (ds_list.__class__ is dict):
        dl = ds_dict_to_list(ds_list)
    
    indexes = np.arange(len(dl))    
    coeffs = []
    for tup in itertools.combinations(indexes, 2):
        
        data1 = dl[tup[0]]
        data2 = dl[tup[1]]
        coeffs.append( average_pearsons_r(data1, data2) )
    
    return np.mean(coeffs)
    
    
    
    
'''====================================================================================
    ***********WARNING TAKES A VERY LONG TIME.  ~9 hours for full dataset******** 
    
    Get the voxel_wise Pearson's correlation coefficient between two subjects for all
    possible pair combinations in the list of Datasets.
    
    ds_list      The list or dictionary of Dataset objects containing subject data. 
                        
    Returns      The mean value of all of the pair combinations of Pearson's |r| values
======================================================================================'''    
def voxelwise_pearsons_ISC(ds_list):
    
    dl = ds_list
    if (ds_list.__class__ is dict):
        dl = ds_dict_to_list(ds_list)
        
    indexes = np.arange(len(dl))   
    coeffs = []
    voxel_coeffs = []
    tuple_combos = list(itertools.combinations(indexes, 2))
    
    #for each voxel position in the subject
    for i in range(dl[0].shape[1]):  
        print(i)        
        voxel_coeffs = []        
        for tup in tuple_combos:        
            data1 = dl[tup[0]][:,i:i+1]
            data2 = dl[tup[1]][:,i:i+1]
            #print(pearsonr(data1, data2)[0])
            voxel_coeffs.append( abs(pearsonr(data1, data2)[0]) )            
      
        coeffs.append(np.mean(voxel_coeffs))        
     
    return coeffs
    
    

'''====================================================================================
    Takes n datasets with v voxels and t time samples each, and creates a numpy array 
    with shape (n, v, t) and inserts it into a new Dataset object with the same voxel
    indices as before. 
    
    ds_list      The list of Dataset objects containing subject data. 
                        
    Returns      A new Dataset object with all of the datasets combined
======================================================================================'''     
def combine_datasets(dslist):
    
    num_samples = dslist[0].shape[0]
    num_voxels = dslist[0].shape[1]
    combined_ds = []
    for row, subject in enumerate(dslist):
        combined_ds.append(np.zeros((num_voxels,num_samples)))
        data = np.transpose(subject.samples)
        for column, voxel in enumerate(data):
            combined_ds[row][column] = voxel
    ds = Dataset(np.array(combined_ds))
    ds.a.mapper = dslist[0].a.mapper
    ds.fa["voxel_indices"] = dslist[0].fa.voxel_indices
    ds.sa.clear()
    ds.sa["subject"] = np.arange(len(dslist))       
    return ds



# TODO: Need to implement multiple measures which take Datasets of shape
# TODO: (n, v, t) and output a scalar correlation between the n subjects
def fake_measure(ds):
    print(ds.shape)
    return 2
    
'''====================================================================================
    Takes a dataset of shape (n, v, t) where n is number of subjects, v is number
    of voxels, and t is number of time samples for each subject.  Runs a parallel
    searchlight analysis on all of the subjects given the metric input.
    
    ds          The Dataset object containing all of the subjects runs
                        
    Returns      The results of the searchlight
======================================================================================''' 
def run_searchlight(ds, metric='correlation', radius=3, nproc=None):
    
    sl = sphere_searchlight(fake_measure, radius=radius, nproc=nproc)
    
    return sl(ds)
    
    

    
    

