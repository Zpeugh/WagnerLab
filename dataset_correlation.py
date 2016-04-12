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
from sklearn.cross_decomposition import CCA


def pearson_r(data1, data2):
    return abs(pearsonr(data1, data2)[0])

def get_metric(name):
    name = name.lower()
    if name == 'pearson':
        return pearson_r


def single_voxel_correlation(index, tuple_combos, datasets, corr_measure):
    voxel_coeffs = 0
    for tup in tuple_combos:        
        data1 = datasets[tup[0]][:,index:index+1]
        data2 = datasets[tup[1]][:,index:index+1]
        voxel_coeffs += corr_measure(data1, data2)
   
    return (index,voxel_coeffs / float(len(tuple_combos)),)
 
 
def voxel_correlations(arg_list):
    return single_voxel_correlation(*arg_list)




def dataset_correlation(ds_list, metric='pearson', num_threads=40, normalize=False):    
           

    indexes = np.arange(len(ds_list))
    datasets = ds_list
    tuple_combos = list(itertools.combinations(indexes, 2))
    
    corr_measure = get_metric(metric)
    
    pool = Pool(num_threads)
    args_list = []
    for index in range(datasets[0].shape[1]):        
        args_list.append( (index, tuple_combos, datasets, corr_measure,) )           
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
    
            
            
            
            
def searchlight(ds_list, metric='pearson', radius=3):
                
    sl = sphere_searchlight(process, radius=radius, space='voxel_indices')   
    niftiresults = map2nifti(sl_map, imghdr=dataset.a.imghdr)   
    
    fig = pl.figure(figsize=(12, 4), facecolor='white')
    subfig = plot_lightbox(overlay=niftiresults, 
                            fig=fig, **plot_args)
    pl.title('Accuracy distribution for radius %i' % radius)

        
            
            
            
def get_average_dataset(ds_list):

    sums = ds_list[0].samples
    num_subjects = len(ds_list)

    for i in range (1, num_subjects):
        sums += ds_list[i].samples      
            
    ds = ds_list[0]

    ds.samples = sums / float(num_subjects)
    return ds       
            
            
            
def isc(ds_list):

    num_voxels = ds_list[0].shape[0]
    avg_ds = get_average_dataset(ds_list)
    
    voxels_coeffs = []   

    for voxel in range(num_voxels):    
        print(voxel)
        voxel_coeffs = 0        
        for ds in ds_list:
            data1 = avg_ds[:,voxel:voxel+1]
            data2 = ds[:,voxel:voxel+1]
            voxel_coeffs += pearson_r(data1, data2)
        voxels_coeffs.append( voxel_coeffs / float(num_voxels) )            
            
    ds = ds_list[0]
    ds.sa.clear  
    ds.samples = np.array(voxels_coeffs)
    
    return ds



            
#def intersubject_corrs(ds, ls):
    #PDistConsistency(center_data=True, pairwise_metric=metric, chunks_attr='subject')

    


            
            