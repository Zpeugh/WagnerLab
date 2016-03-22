# -*- coding: utf-8 -*-


from mvpa2.tutorial_suite import *
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np

#####################################OLD SCRIPT TO LOAD DATA#######################




########################## GLOBAL VARIABLES FOR 2010 DATA ###########################
INCORRECT_SAMPLE_RATE = 2.512
CORRECT_SAMPLE_RATE = 2.5
TOTAL_SLICED_SAMPLES = 633

MASK_FILE_PATH = 'masks/bigmask_3x3x3.nii'
BASE_PATH = '/lab/neurodata/dartmouth/2010_SP/SUBJECTS/'
RUN_PATH = "/FUNCTIONAL/swuabold{0}.nii"
PARAM_FILE_PATH = "/FUNCTIONAL/rp_abold{0}.txt"


#####################################################################################

## function to resample a dataset using Fourier transforms from a scipy module
## ds is a Dataset object
## old_rample rate is the rate of acquisition in samples/second that was used
## new_sample rate is the desired sample/second rate
def ds_resample( ds, old_sample_rate=INCORRECT_SAMPLE_RATE, new_sample_rate=CORRECT_SAMPLE_RATE ):
 
    original_sample_length = len(ds.samples)
    resampling_number =  (original_sample_length * new_sample_rate ) / old_sample_rate
    transposed_samples = np.transpose( ds.samples )
    resampled_samples = transposed_samples.copy()        
    for i, row in enumerate(transposed_samples):
        # resample then add in the spliced 2 samples as 0s on the end to avoid a dimension mismatch
        resampled_data = ss.resample(row, resampling_number)
        samples_gained = len(resampled_data) - original_sample_length
        
        if samples_gained > -1:
            resampled_samples[i] = resampled_data[:original_sample_length]
        else:
            resampled_samples[i] = np.append(resampled_data, np.zeros(-(samples_gained) ) )
            
    ds.samples = np.transpose(resampled_samples)    
    return ds
    
def z_score_dataset(ds):   
    zscore(ds)
    return ds

def detrend_data_with_design_matrix(runData, paramMatrix):
    ds = runData.copy()
    y = runData.samples
    
    # Insert a row of 1's at the front of the design matrix     
    numParams = paramMatrix.shape[1] + 1
    x = np.ones((paramMatrix.shape[0],numParams))  
    x[:,1:numParams] = paramMatrix

    #X'           
    x_prime = np.transpose(x)
      
    n = np.linalg.inv(np.dot(x_prime, x) )
    m = np.dot(x_prime, y)
    
    #b = ((X'X)^-1)X'y 
    betas = np.dot(n,m)
    #y_hat = Xb
    y_hat = np.dot(x,betas)
    #residuals = y-y_hat  
    ds.samples = y - y_hat
    return ds

    
def combineRuns(run1, run2, run3):
    
    finalDataset = Dataset(np.concatenate( (run1.samples[38:249], run2.samples[38:249], run3.samples[38:249]) ) )
    finalDataset.fa["voxel_indices"] = run1.fa.voxel_indices
    finalDataset.sa["chunks"] = np.concatenate( (np.zeros(211), np.zeros(211) + 1, np.zeros(211) + 2) )
    finalDataset.sa["time_indices"] = np.arange(TOTAL_SLICED_SAMPLES)
    finalDataset.sa["time_coords"] = np.arange(0,TOTAL_SLICED_SAMPLES * CORRECT_SAMPLE_RATE, CORRECT_SAMPLE_RATE)    

    return finalDataset    




##The list of subject file names with the proper sampling rate of 2.5 s/sec
subjects = ['0ctr_14oct09ft'] #, '0ctr_30sep09kp', '0smk_17apr09ag', '0ctr_30sep09ef'] 
#'0ctr_14oct09gl', '0ctr_30sep09sh', '0smk_22apr09cc','0ctr_14oct09js', '0ctr_30sep09so', 
#'0ctr_18apr09yg', '0ctr_30sep09zl', '0ctr_19apr09tj', '0ctr_26jul09bc', '0smk_28sep09cb',
#'0ctr_28sep09kr', '0ctr_28sep09sb', '0ctr_28sep09sg', '0ctr_29sep09ef', '0ctr_29sep09gp', 
#'0ctr_29sep09mb', '0smk_13oct09ad', '0smk_31jul07sc_36slices', '0smk_07aug07lr_36slices',
# '0smk_07jun07nw_36slices'  ]

##The list of subject file names with the improper sampling rate of 2.512 s/sec
subjects_to_resample = ["0smk_02apr08jb"] # , "0smk_06may08md", "0smk_08may08kw"] 
#"0smk_12may08ne", "0smk_12may08sb", "0smk_14mar07jm" "0smk_25feb08rl", 
#"0smk_25feb08rz", "0smk_27feb08mi", "0smk_30may08sm" 

##The dictionary to contain the 34 subjects with 3 runs each
dataset_dict = dict()

##store all of the normal datasets in the dictionary
for index, subject in enumerate(subjects):
    
    # Load in the datasets for each run, adding the mask provided.
    run1 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(1), mask=MASK_FILE_PATH )    
    run2 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(2), mask=MASK_FILE_PATH ) 
    run3 = fmri_dataset( BASE_PATH + subject + RUN_PATH.format(3), mask=MASK_FILE_PATH ) 
  
    # Load in the parameters for each run
    params1 = np.loadtxt( BASE_PATH + subject + PARAM_FILE_PATH.format(1) )
    params2 = np.loadtxt( BASE_PATH + subject + PARAM_FILE_PATH.format(2) )
    params3 = np.loadtxt( BASE_PATH + subject + PARAM_FILE_PATH.format(3) )    
   
    detrended1 = detrend_data_with_design_matrix(run1, params1)
    detrended2 = detrend_data_with_design_matrix(run2, params2)
    detrended3 = detrend_data_with_design_matrix(run3, params3)
        
    dataset_dict[ "subject_{0}".format(index) ] = combineRuns(detrended1, detrended2, detrended3)
    #dataset_dict[ "subject_{0}".format(index) ] = combineRuns(run1, run2, run3)
    
    
    
number_of_subjects = len(subjects)

#Resample and store all of the subjects who need resampling and place in the dataset_dict
for index, subject in enumerate(subjects_to_resample):  
  
    run1 = ds_resample( fmri_dataset( BASE_PATH + subject + RUN_PATH.format(1), mask=MASK_FILE_PATH ) )
    run2 = ds_resample( fmri_dataset( BASE_PATH + subject + RUN_PATH.format(2), mask=MASK_FILE_PATH ) )
    run3 = ds_resample( fmri_dataset( BASE_PATH + subject + RUN_PATH.format(3), mask=MASK_FILE_PATH ) )    
    
    # Load in the parameters for each run
    params1 = np.loadtxt( BASE_PATH + subject + PARAM_FILE_PATH.format(1) )
    params2 = np.loadtxt( BASE_PATH + subject + PARAM_FILE_PATH.format(2) )
    params3 = np.loadtxt( BASE_PATH + subject + PARAM_FILE_PATH.format(3) )    
   
    detrended1 = detrend_data_with_design_matrix(run1, params1)
    detrended2 = detrend_data_with_design_matrix(run2, params2)
    detrended3 = detrend_data_with_design_matrix(run3, params3)
  
    dataset_dict[ "subject_{0}".format(index) ] = combineRuns(detrended1, detrended2, detrended3)
    dataset_dict[ "subject_{0}".format(index + number_of_subjects) ] = combineRuns(run1, run2, run3)    
 

 
 
def voxel_plot(ds, voxel_position):
    ts = np.transpose(ds.samples)
    
    plt.figure(figsize=(10,6))
    plt.plot(ts[voxel_position])
    plt.title("Timeseries for voxel {0}".format(ds.fa.voxel_indices[voxel_position]))
    plt.axvline(211, color='r', linestyle='--')
    plt.axvline(422, color='r', linestyle='--')
    plt.show




   
    
   