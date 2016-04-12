# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:45:57 2016

@author: peugh.14
"""

import numpy as np
import matplotlib.pyplot as plt
from mvpa2.tutorial_suite import Dataset
import fmri_preprocessing as fp
import time
from multiprocessing import Pool
from scipy.stats import pearsonr

class DatasetCorrelation:
    
    
    datasets = []
    
    def get_metric(name):
        name = name.lower()
        if name == 'pearson'
            return pearsonr
        
            
    
    
    def __init__(self, ds_list):
        self.datasets = ds_list
        
    
    
    
    
    
    
    def intersubject_correlation(ds_list, metric='Pearson', num_threads=40):
        pool = Pool(num_threads)
    
            
        indexes = np.arange(len(dl))   
        coeffs = []
        voxel_coeffs = []
        tuple_combos = list(itertools.combinations(indexes, 2))
        
        #for each voxel position in the subject
        for i in range(dl[0].shape[1]):  
            print(i)        
            voxel_coeffs = []        
            for tup in tuple_combos:        
                data1 = ds_list[tup[0]][:,i:i+1]
                data2 = ds_list[tup[1]][:,i:i+1]
                voxel_coeffs.append( abs(pearsonr(data1, data2)[0]) )            
          
            coeffs.append(np.mean(voxel_coeffs))        
         
        return coeffs
            