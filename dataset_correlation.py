# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:45:57 2016

@author: peugh.14
"""
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import pearsonr
from mvpa2.tutorial_suite import Dataset
from operator import itemgetter


def pearson_r(data1, data2):
    return abs(pearsonr(data1, data2)[0])

def get_metric(name):
    name = name.lower()
    if name == 'pearson':
        return pearson_r


def single_voxel_correlation(index, tuple_combos, datasets, metric):
    voxel_coeffs = []
    corr_measure = get_metric(metric)
    for tup in tuple_combos:        
        data1 = datasets[tup[0]][:,index:index+1]
        data2 = datasets[tup[1]][:,index:index+1]
        voxel_coeffs.append( corr_measure(data1, data2) )
   
    return (index, np.mean(voxel_coeffs))
 
 
def voxel_correlations(arg_list):
    return single_voxel_correlation(*arg_list)




def dataset_correlation(ds_list, metric='pearson', num_threads=40, normalize=False):    
           

    indexes = np.arange(len(ds_list))
    datasets = ds_list
    tuple_combos = list(itertools.combinations(indexes, 2))
    
    
    pool = Pool(num_threads)
    args_list = []
    for index in range(datasets[0].shape[1]):        
        args_list.append( (index, tuple_combos, datasets, metric,) )           
    t_0 = time.time()
    results = pool.map(voxel_correlations, args_list)
    t_elapsed = time.time() - t_0 
    
    print("Time elapsed: %.4f" % t_elapsed)
    
    ds = ds_list[0].copy()    
    ds.sa.clear    
    ordered_coeffs = sorted(results,key=itemgetter(0))
     
    ds.samples = np.array([x[1] for x in ordered_coeffs])    
    
    if normalize:
        #normalize samples for better visualization
        ds.samples = ds.samples / np.max(ds.samples)    
    
    return ds
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            