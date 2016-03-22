# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:19:35 2016

@author: peugh.14
"""


# -*- coding: utf-8 -*-


from mvpa2.tutorial_suite import *
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np


########################## GLOBAL VARIABLES FOR 2010 DATA ###########################
INCORRECT_SAMPLE_RATE = 2.512
CORRECT_SAMPLE_RATE = 2.5
TOTAL_SLICED_SAMPLES = 633

MASK_FILE_PATH = 'masks/bigmask_3x3x3.nii'
BASE_PATH = '/lab/neurodata/dartmouth/2010_SP/SUBJECTS/'
END_1 = '/FUNCTIONAL/swuabold1.nii'
END_2 = '/FUNCTIONAL/swuabold2.nii'
END_3 = '/FUNCTIONAL/swuabold3.nii'    
#####################################################################################


detrender = PolyDetrendMapper(polyord=3, chunks_attr='chunks')

## function to resample a dataset using Fourier transforms from a scipy module
## ds is a Dataset object
## old_rample rate is the rate of acquisition in samples/second that was used
## new_sample rate is the desired sample/second rate
def ds_resample( ds, old_sample_rate=INCORRECT_SAMPLE_RATE, new_sample_rate=CORRECT_SAMPLE_RATE ):
 
    resampling_number =  ( len(ds.samples) * new_sample_rate ) / old_sample_rate
    transposed_samples = np.transpose( ds.samples )
    resampled_samples = transposed_samples.copy()
    i = 0        
    for row in transposed_samples:
        # resample then add in the spliced 2 samples as 0s on the end to avoid a dimension mismatch
        resampled_samples[i] = np.append( ss.resample(row, resampling_number), [0,0] ) 
        i += 1
    ds.samples = np.transpose(resampled_samples)    
    return ds
    
    
def combineRuns(run1, run2, run3):
    
    finalDataset = Dataset(np.concatenate( (run1.samples[38:249], run2.samples[38:249], run3.samples[38:249]) ) )
    finalDataset.fa["voxel_indices"] = run1.fa.voxel_indices
    finalDataset.sa["chunks"] = np.concatenate( (np.zeros(211), np.zeros(211) + 1, np.zeros(211) + 2) )
    finalDataset.sa["time_indices"] = np.arange(TOTAL_SLICED_SAMPLES)
    finalDataset.sa["time_coords"] = np.arange(0,TOTAL_SLICED_SAMPLES * CORRECT_SAMPLE_RATE, CORRECT_SAMPLE_RATE)
    finalDataset = finalDataset.get_mapped(detrender)
    #zscore(finalDataset)

    return finalDataset    

##The list of subject file names with the proper sampling rate of 2.5 s/sec
subjects = ['0ctr_14oct09ft']
#, '0ctr_30sep09kp','0smk_17apr09ag', '0ctr_30sep09ef'] 
#'0ctr_14oct09gl', '0ctr_30sep09sh', '0smk_22apr09cc','0ctr_14oct09js', '0ctr_30sep09so', 
#'0ctr_18apr09yg', '0ctr_30sep09zl', '0ctr_19apr09tj', '0ctr_26jul09bc', '0smk_28sep09cb',
#'0ctr_28sep09kr', '0ctr_28sep09sb', '0ctr_28sep09sg', '0ctr_29sep09ef', '0ctr_29sep09gp', 
#'0ctr_29sep09mb', '0smk_13oct09ad', '0smk_31jul07sc_36slices', '0smk_07aug07lr_36slices',
# '0smk_07jun07nw_36slices'  ]

##The list of subject file names with the improper sampling rate of 2.512 s/sec
subjects_to_resample = ["0smk_02apr08jb", "0smk_06may08md", "0smk_08may08kw"] 
#"0smk_12may08ne", "0smk_12may08sb", "0smk_14mar07jm" "0smk_25feb08rl", 
#"0smk_25feb08rz", "0smk_27feb08mi", "0smk_30may08sm" 

##The dictionary to contain the 34 subjects with 3 runs each
dataset_dict = dict()

##store all of the normal datasets in the dictionary
for index, subject in enumerate(subjects):
    run1 = fmri_dataset( BASE_PATH + subject + END_1, mask=MASK_FILE_PATH )
    run2 = fmri_dataset( BASE_PATH + subject + END_2, mask=MASK_FILE_PATH )
    run3 = fmri_dataset( BASE_PATH + subject + END_3, mask=MASK_FILE_PATH )
   
    dataset_dict[ "subject_{0}".format(index) ] = combineRuns(run1, run2, run3)
    
    
number_of_subjects = len(subjects)

##Resample and store all of the subjects who need resampling and place in the dataset_dict
#for index, subject in enumerate(subjects_to_resample):    
#    run1 = ds_resample( fmri_dataset( BASE_PATH + subject + END_1, mask=MASK_FILE_PATH ) )
#    run2 = ds_resample( fmri_dataset( BASE_PATH + subject + END_2, mask=MASK_FILE_PATH ) )
#    run3 = ds_resample( fmri_dataset( BASE_PATH + subject + END_3, mask=MASK_FILE_PATH ) )
#    
#    run1 = fmri_dataset( BASE_PATH + subject + END_1, mask=MASK_FILE_PATH )
#    run2 = fmri_dataset( BASE_PATH + subject + END_2, mask=MASK_FILE_PATH )
#    run3 = fmri_dataset( BASE_PATH + subject + END_3, mask=MASK_FILE_PATH )
#    
#    
#    dataset_dict[ "subject_{0}".format(index + number_of_subjects) ] = combineRuns(run1, run2, run3)
#    
 
#############################Append 3 trial runs together for each subject#################

def voxel_plot(ds, voxel_position):
    ts = np.transpose(ds.samples)
    
    plt.figure(figsize=(10,6))
    plt.plot(ts[voxel_position])
    plt.title("Timeseries for voxel {0}".format(ds.fa.voxel_indices[voxel_position]))
    plt.axvline(211, color='r', linestyle='--')
    plt.axvline(422, color='r', linestyle='--')
    plt.show













   
    
   