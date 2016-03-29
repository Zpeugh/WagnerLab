# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:05:58 2016

@author: Zach Peugh
"""

import numpy as np
from mvpa2.tutorial_suite import *
import matplotlib.pyplot as plt
import scipy.signal as ss

################################Utility file for preprocessing data######################



'''=====================================================================================
    Function to resample a dataset using Fourier transforms from a scipy module

    ds                the Dataset object
    old_rample rate   the rate of acquisition in samples/second that was used
    new_sample rate   the desired sample/second rate

    returns           the resampled Dataset object.  Fills the end of the dataset with 
                      zeros if the new sampling rate is lower than the old sampling rate
======================================================================================'''
def ds_resample( ds, old_sample_rate, new_sample_rate ):
 
    original_sample_length = len(ds.samples)
    resampling_number =  (original_sample_length * new_sample_rate ) / old_sample_rate
    transposed_samples = np.transpose( ds.samples )
    resampled_samples = transposed_samples.copy()
        
    for i, row in enumerate(transposed_samples):
        resampled_data = ss.resample(row, resampling_number)
        samples_gained = len(resampled_data) - original_sample_length        
        if samples_gained > -1:
            resampled_samples[i] = resampled_data[:original_sample_length]
        else:
            resampled_samples[i] = np.append(resampled_data, np.zeros(-(samples_gained) ) )
            
    ds.samples = np.transpose(resampled_samples)    
    return ds
    
    
'''========================================================
    Function to normalize the dataset samples by Z_scoring

    ds        the Dataset object
 
    returns   the z_scored Dataset object
#========================================================'''
def z_score_dataset(ds):   
    zscore(ds)
    return ds


'''=============================================================================
    Appends columns of data to a matrix.  Intended for adding parameters to a
    to a design matrix.

    left        the numpy array of data with shape (n,m)
    right       the numpy array to add the column to.  Has shape (n,p)

    returns     the combined matrix
============================================================================='''
def add_columns_to_matrix(left, right):   
    return np.hstack( (left, right) )    


'''=============================================================================
    Appends rows of data to a matrix.  Intended for adding parameters to a
    to a design matrix.

    left        the numpy array of data with shape (m,n)
    right       the numpy array to add the column to.  Has shape (p,n)

    returns     the combined matrix
============================================================================='''
def add_rows_to_matrix(top, bottom):
    return np.vstack( (top, bottom) )
    
    

'''=============================================================================
    Fits polynomials from degree 1 to n and appends them all to into a design matrix.

    num_samples  the number of samples in the data
    degrees      the number of degrees to filter out from 1 to 'degrees' inclusive

    returns      the polynomial design matrix
==============================================================================='''
def get_polynomial_design_matrix(num_samples, degrees=4):
   
    x = np.arange(num_samples)
    matrix = 0
    matrix_set = False
    y = np.zeros((num_samples,degrees))
    
    for i in range(1,degrees+1):       
        y[:,i-1] = np.power(np.linspace(1, -1, num_samples), i)        
        
    return y


'''=================================================================================
    Takes a design matrix and the raw data and detrends it using multiple regression 

    ds                Dataset object with data matrix of shape (n,m)
    design_matrix     numpy array with design matrix of shape (n,p)

    returns           the detrended dataset with the same shape (n,p) as the original
==================================================================================='''
def detrend_data_with_design_matrix(ds, design_matrix):
    
    raw_data = ds.samples                               #y = data 
    x_prime = np.transpose(design_matrix)               #X' = X^T
    n = np.linalg.inv(np.dot(x_prime, design_matrix) )  #n = X'X
    m = np.dot(x_prime, raw_data)                       #m = X'y

    betas = np.dot(n,m)                                 #b = ((X'X)^-1)X'y
    y_hat = np.dot(design_matrix,betas)                 #y_hat = Xb
    
    ds.samples = raw_data - y_hat                       #residuals = y-y_hat     

    return ds             


'''=================================================================================
    Gives the properly stacked configuration of polynomial paramaters for a design
    matrix given the number of samples and runs

    samples_per_run     The number of samples per run
    num_runs            The number of runs
    num_degrees         The number of polynomial degrees to add to the matrix

    returns             The polynomial parameter matrix for the runs
==================================================================================='''
def polynomial_matrix(samples_per_run, num_runs, num_degrees): 

    polynomial_matrix = get_polynomial_design_matrix(samples_per_run, degrees=num_degrees)
    design_matrix = np.zeros( (samples_per_run * num_runs , num_degrees * num_runs) )    
   
    for i in range(num_runs):        
        r_start = i * samples_per_run
        c_start = i * num_degrees
        r_end = r_start + samples_per_run
        c_end = c_start + num_degrees        
    
        design_matrix[r_start:r_end, c_start:c_end] = polynomial_matrix 
    
    return design_matrix
        

'''=================================================================================
    Gives the properly stacked configuration of constant paramaters for a design
    matrix given the number of samples and runs

    samples_per_run     The number of samples per run
    num_runs            The number of runs    

    returns             num_runs - 1 columns of properly oriented constant parameters
==================================================================================='''
def constant_matrix(samples_per_run, num_runs):
    
    c_matrix = np.ones((samples_per_run, 1))
    design_matrix = np.zeros( (samples_per_run * num_runs , num_runs - 1) )
    
    for i in range(num_runs):
        r_start = i * samples_per_run     
        r_end = r_start + samples_per_run    
        design_matrix[r_start:r_end, i:i + 1] = c_matrix 
    
    return design_matrix
    

'''=================================================================================
    Takes a list of motion parameter matrices, read in from the FMRI machine output
    and returns them stacked vertically

    param_matrices      The list of numpy array motion parameter matrices

    returns             The vertically concatenated matrix of motion parameters
==================================================================================='''
def param_matrix(param_matrices):
    
    param_matrix = param_matrices[0]  
    num_runs = len(param_matrices)
     
    for i in range(1, num_runs):
        param_matrix = add_rows_to_matrix(param_matrix, param_matrices[i])    
        
    return param_matrix

    
    
'''====================================================================================
    Takes a parameter matrix from the machine output and adds polynomials to complete
    the design matrix
  
    list_of_param_matrices      List of numpy array of parameter matrix from fmri  
                                machinereadouts for the data.  Should have 
                                shape (n, 6) where n is the number of samples
    num_degrees                 The number of polynomial degrees to consider in the 
                                design matrix.
 
    returns                     the full design matrix
======================================================================================'''
def get_design_matrix(list_of_param_matrices, num_degrees):
    
    samples_per_run = list_of_param_matrices[0].shape[0] 
    col_split = num_degrees + list_of_param_matrices[0].shape[1]
    num_runs = len(list_of_param_matrices)
    
    poly_matrix = polynomial_matrix(samples_per_run, num_runs, num_degrees)
    c_matrix = constant_matrix(samples_per_run, num_runs)         
    design_matrix = add_columns_to_matrix(poly_matrix, c_matrix)    
    p_matrix = param_matrix(list_of_param_matrices)    
    design_matrix = add_columns_to_matrix(design_matrix, p_matrix)     
    design_matrix = add_columns_to_matrix(design_matrix, np.ones( (samples_per_run * num_runs,1) ))

  
    return design_matrix
    

'''======================================================================================
   Takes a Numpy array and returns the data between the beginning and end offset marks.
 
   samples       The numpy array of data samples to be sliced
   beg_offset    (Optional) The number of samples to exclude at the beginning of the run
   end_offset    (Optional) The number of samples to exclude from the end of the run.  
   symmetric     (Optional) False by default.  But if true then end_offset will be
               equal to beg_offset, splicing an equal number of runs from the both sides

   returns      spliced samples
======================================================================================'''
def spliceRuns(samples, beg_offset=0, end_offset=0, symmetric=False):
    sample_size = len(samples)
    if symmetric:
        end_offset = beg_offset
    return samples[beg_offset: sample_size-end_offset]    
    


'''====================================================================================
    Takes a list of Dataset objects with fmri data and combines the runs.
  
    ds_list         The list of Dataset objects to be combined
    sample_rate     (Optional) the rate in samples/second at which the data was collected
                    This is used in computing the time_coords.
                    
    returns         the detrended dataset with voxel_indices, time_coords, time_indices
                    and chunks attributes added.
======================================================================================'''
def combineRuns(ds_list, sample_rate=0):
       
    ds_tuple = ()
    chunks_tuple = ()
    run_samples = ds_list[0].shape[0]
    
    for i, ds in enumerate(ds_list):
        ds_tuple = ds_tuple + ( ds,  )
        chunks_tuple = chunks_tuple + ( (np.zeros(run_samples) + i), )
        
    combined_ds = Dataset( np.concatenate(ds_tuple) ) 
        
    total_samples = combined_ds.shape[0]
        
    combined_ds.fa["voxel_indices"] = ds_list[0].fa.voxel_indices         
    combined_ds.sa["chunks"] = np.concatenate(chunks_tuple)
    combined_ds.sa["time_indices"] = np.arange(total_samples)
    if( sample_rate > 0 ):
       combined_ds.sa["time_coords"] = np.linspace(0,sample_rate * (total_samples-1), total_samples )
    
    return combined_ds      



'''====================================================================================
    Takes a Dataset object and throws out all sections of runs that are unwanted,
    returning only the desired, spliced data.
  
    ds          The Dataset containing runs of data which need samples thrown out
    num_runs    The number of runs concatenated together in ds
    beg_offset  (optional) The number of samples to throw away at the front of each run
    end_offset  (optional) The number of samples to throw away at the end of each run
                    
    returns     The Dataset with unwanted samples thrown out. 
======================================================================================'''
def splice_ds_runs(ds, num_runs, beg_offset=0, end_offset=0):
      
    orig_run_length = ds.samples.shape[0] / num_runs  
    new_run_length = orig_run_length - beg_offset - end_offset
    sliced_samples = np.zeros((new_run_length * num_runs, ds.samples.shape[1])) 
    sliced_chunks = np.zeros(new_run_length * num_runs)
    sliced_t_coords = sliced_chunks.copy()
    sliced_t_indices = sliced_chunks.copy()    
    
    for i in range(num_runs):
        o_start = i * orig_run_length
        n_start = i * new_run_length
        
        samples = ds.samples[o_start:o_start + orig_run_length, :]
        chunks = ds.sa.chunks[o_start:o_start + orig_run_length]
        t_coords = ds.sa.time_coords[o_start:o_start + orig_run_length]
        t_indices = ds.sa.time_indices[o_start:o_start + orig_run_length]
           
        new_samples = samples[beg_offset: orig_run_length-end_offset, :]
        new_chunks = chunks[beg_offset: orig_run_length-end_offset]
        new_t_coords = t_coords[beg_offset: orig_run_length-end_offset]        
        new_t_indices= t_indices[beg_offset: orig_run_length-end_offset]
        
        sliced_samples[n_start:n_start + new_run_length,:] = new_samples
        sliced_chunks[n_start:n_start + new_run_length] = new_chunks
        sliced_t_coords[n_start:n_start + new_run_length] = new_t_coords
        sliced_t_indices[n_start:n_start + new_run_length] = new_t_indices
    
    new_ds = Dataset(sliced_samples)    
    new_ds.sa["chunks"] = sliced_chunks
    new_ds.sa.time_coords = sliced_t_coords
    new_ds.sa.time_indices = sliced_t_indices

    return new_ds     




##############################OLD #############################################
def old_get_design_matrix(list_of_param_matrices, num_degrees):

    row_split = list_of_param_matrices[0].shape[0] 
    col_split = num_degrees + list_of_param_matrices[0].shape[1]
    num_runs = len(list_of_param_matrices)
   
    constant_column = np.ones( (row_split,1) )
    polynomial_matrix = get_polynomial_design_matrix(row_split, degrees=num_degrees)
    

    design_matrix = np.zeros( (row_split * num_runs , (col_split + 1) * num_runs) )    
        
    for i, param_matrix in enumerate(list_of_param_matrices):
        
        r_start = i * row_split
        c_start = i * (col_split + 1)
        r_end = r_start + row_split
        c_end = c_start + col_split + 1
        run_matrix = add_columns_to_matrix(polynomial_matrix, param_matrix)
    
        design_matrix[r_start:r_end, c_start:c_end] = add_columns_to_matrix(run_matrix, constant_column)
    
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.imshow(design_matrix, cmap='gray', aspect='auto', interpolation='none')
    plt.show()
    
  
    return design_matrix
    
    
    











