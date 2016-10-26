# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:46:17 2016

@author: peugh.14
"""


import load_data as ld
import dataset_utilities as du
import pickle
import matplotlib.pyplot as plt
import time
import timeit
from mvpa2.suite import *
import numpy as np
import scipy.stats as stats

PEARSON_RADIUS_2_P_THRESH = 0.1526
CCA_RADIUS_2_P_THRESH = 0.384783972034



'''=======================================================================================
    Run intersubject correlation (Pearson's r) and intersubject information 
    (canonical correlation analysis) on the Haxby_2010 dataset and save a pickle of the
    resulting Dataset for each, along with nifti files and the graph of ISC vs ISI.
    
    radius           the radius of the searchlight in voxels
    n_cpu            the number of cpu cores to run on
    subjects         number of subjects to test on
    brain_region     the string prefix of files to explain what region of the brain
                     is being tested.  i.e. 'left_hemisphere'
    mask_path        the path to the .nii mask for the corresponding brain region
=======================================================================================''' 
def run_cca_and_isc(radius, n_cpu, subjects, brain_region, mask_path):
    
    # Get the datasets
    cds = ld.get_2010_preprocessed_data(num_subjects=subjects, mask_path=mask_path, n_cpu=n_cpu)
    # Run CCA and save results to pickle and nifti file
    cca_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='cca')
    f = open('results/data/{0}_cca_{1}_{2}.pckl'.format(brain_region, subjects, radius), 'wb')
    pickle.dump(cca_res, f)
    f.close()   
    du.export_to_nifti(cca_res, 'results/nifti/{0}_cca_{1}s_r{2}'.format(brain_region,subjects, radius))

    # Run Pearson's correlation and save results to pickle and nifti file    
    corr_res = du.run_searchlight(cds, n_cpu=n_cpu, radius=radius, metric='correlation')
    f = open('results/data/{0}_corr_{1}_{2}.pckl'.format(brain_region, subjects, radius), 'wb')
    pickle.dump(corr_res, f)
    f.close()
    du.export_to_nifti(corr_res, 'results/nifti/{0}_corr_{1}s_r{2}'.format(brain_region,subjects, radius))
    
    # Plot the colored and ISC vs ISI plot
    du.plot_colored_isc_vs_isi(corr_res, cca_res, voxels=corr_res.fa.voxel_indices, title='{0} {1} Subject ISI vs ISC: Searchlight Radius {2}'.format(brain_region, subjects, radius), save=True, filename='results/figures/{0}_{1}_{2}'.format(brain_region, subjects, radius))    

   
    

def pvalues(num_subjects=34, radius=3, mask_path='masks/bigmask_3x3x3.nii', n_cpu=50):

    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
    res = du.run_searchlight(cds, metric='pvalues', radius=radius, n_cpu=n_cpu)
    
    f = open('results/data/full_brain_p_values_r3.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    
    return res



def tvalues(num_subjects=34, radius=3, mask_path='masks/bigmask_3x3x3.nii', n_cpu=50):

    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
    res = du.run_searchlight(cds, metric='tvalues', radius=radius, n_cpu=n_cpu)
    
    f = open('results/data/full_brain_t_values_r2.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    
    return res
    

def timing_test(metric='cca', n_cpu=20, filename=None):

    cds = ld.get_2010_preprocessed_data(mask_path='masks/aal_l_ifg_oto_3x3x3.nii')
    
    times = []     
    for rad in range(7):
        t_0 = time.time()        
        du.run_searchlight(cds, radius=rad, n_cpu=n_cpu)
        times.append((time.time()-t_0) * n_cpu)
    [1,5.3,20,53,86]
    plt.clf()
    fig = plt.figure(figsize=(10,6))
    plt.plot(times, '-o')
    plt.xlabel('Searchlight Radius (voxels)')
    plt.ylabel('Single Core Time (seconds)')
    plt.title('CCA Time as a Function of Searchlight Radius')
    if filename:
       fig.savefig(filename) 
    plt.show()
    
    return times
    
def compare_transportation_groups(cds, high=5.5, low=3, metric='correlation', radius=2, n_cpu=20):
    
    num_samples = cds.shape[2]
    num_voxels = cds.shape[1]
    
    high_indices = [i for i, x in enumerate(cds.sa.transportation) if x>=high]
    low_indices = [i for i, x in enumerate(cds.sa.transportation) if x<=low]
    
    print("\n{0} subjects have scores above {1}".format(len(high_indices), high))
    print("{0} subjects have scores below {1}".format(len(low_indices), low))    
    
    # Get the Dataset for subjects with high 'transportation' scores
    high_tup = ()
    for i in high_indices:
        high_tup = high_tup + (cds.samples[i, :, :],)
    
    high_ds = Dataset(np.vstack(high_tup).reshape((len(high_indices), num_voxels, num_samples)))    
    high_ds.a.mapper = cds.a.mapper
    high_ds.fa["voxel_indices"] = cds.fa.voxel_indices    
   
    # Get the Dataset for subjects with low 'transportation' scores
    low_tup = ()
    for i in low_indices:
        low_tup = low_tup + (cds.samples[i, :, :],)
    
    low_ds = Dataset(np.vstack(low_tup).reshape((len(low_indices), num_voxels, num_samples)))    
    low_ds.a.mapper = cds.a.mapper
    low_ds.fa["voxel_indices"] = cds.fa.voxel_indices


    high_res = du.run_searchlight(high_ds, radius=radius, n_cpu=n_cpu, metric=metric)
    low_res = du.run_searchlight(low_ds, radius=radius, n_cpu=n_cpu, metric=metric)
    
    
    print("High transportation average {0} value: {1}".format(metric, high_res.samples.mean()))    
    print("Low transportation average {0} value: {1}".format(metric, low_res.samples.mean()))   
    
    return high_res, low_res
    


    
def test_all_pairs(cds, radius=2, n_cpu=20, metric='correlation', method='multiply', feature='transportation'):
    
   
    X = du.pairwise_feature_list(cds, feature=feature, method=method)
    
    if metric == 'correlation':
        res = du.run_searchlight(cds, radius=radius, n_cpu=n_cpu, metric='all_pearsons')
    elif metric == 'cca':
        res = du.run_searchlight(cds, radius=radius, n_cpu=n_cpu, metric='all_cca')
    else:
        print("INVALID METRIC.  Choose 'correlation' or 'cca'")
    means = np.mean(res.samples, axis=1)   
    
    plt.plot(X, means, 'x')
    
    return res
    
    
def plot_feature_vs_result(cds, means, feature, title, method='multiply'):

    X = du.pairwise_feature_list(cds, feature, method=method)
    
    plt.plot(X, means, 'o')
    plt.xlabel(feature)   
    plt.ylabel('Correlation')
    plt.title(title)
    plt.show()
    
    print("Correlation: {0}".format(np.corrcoef(X, means)[0,1]))

def plot_activation_vs_scene_change(mask_path, window=5, a=0.01, n_cpu=20):

    scenes = ld.get_2010_scene_splits()
    cds = ld.get_2010_preprocessed_data(mask_path=mask_path, n_cpu=n_cpu)
    res = du.run_searchlight(cds, metric="tvalues", n_cpu=n_cpu)
    du.plot_activation_with_scenes(res, scenes, window=window, a=a, n=cds.shape[0])

    
    
        
    
def validation_cca(num_subjects, mask_path, radii=[0,1,2], n_cpu=None):
    
    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path)
    
      
    cancorrs = []
    max_corrs = []
    results = dict()
    for rad in radii:
        t_0 = time.time()
        cca_val_res = du.run_searchlight(cds, metric='rcca_validate_max', radius=rad, n_cpu=n_cpu)
        cca_res = du.run_searchlight(cds, metric='cca', radius=rad, n_cpu=n_cpu)
        results["radius_{0}".format(rad)] = cca_val_res        

        t_elapsed = time.time() - t_0
        print("Done with radius {0}\nTook {1} seconds".format(rad, t_elapsed))
        du.plot_colored_isc_vs_isi(cca_res, cca_val_res, "Validated CCA vs. Max Correlation: Radius {0}".format(rad), xlabel="CCA correlation", ylabel="CCA Validation Predicted Correlation")


    return results    
    
def pick_random_pair_shift_one(cds):
    num_subj = cds.shape[0]

    s1 = np.random.randint(0,num_subj)
    s2 = s1    
    while s1==s2:
        s2 = np.random.randint(0,num_subj)
        
    subj1 = cds.samples[s1,:,:]
    
    samples = np.zeros((2, cds.shape[1], cds.shape[2]))
    samples[0,:,:] = subj1
    #samples[1,:,:] = du.randomize_subject(cds.samples[s2,:,:])
    samples[1,:,:] = du.shift_subject(cds.samples[s2,:,:])
    ds = Dataset(samples) 
    ds.a = cds.a
    ds.fa = cds.fa
    
    return ds
    
    
def random_shift_all_but_n(cds, n):
    num_subj = cds.shape[0]  
    print("shifting all but subject:", n)
    ds = cds.copy()
    temp = cds.samples[0,:,:]
    ds.samples[0, :, :] = cds.samples[n,:,:]
    ds.samples[n,:,:] = temp
    for i in range(1,num_subj):
        ds.samples[i,:,:] = du.shift_subject(cds.samples[i,:,:])
            
    return ds    
    
def create_null_p_mapping(cds, n=10, radius=2, alpha = 0.05, n_cpu=10):
    
    results = dict() 
    results["cca"] = []
    results["pearson"] = []   
    
    for i in range(0,n):
        ds = random_shift_all_but_n(cds, i % cds.shape[0])
        cca_res = du.run_searchlight(ds, metric="1_to_many_cca", radius=radius, n_cpu=n_cpu)
        #corr_res = du.run_searchlight(ds, metric="correlation", radius=radius, n_cpu=n_cpu)
        results["cca"].append(np.mean(cca_res.samples))
        #results["pearson"].append(np.mean(corr_res.samples))
        if (i % 2 == 0):
            #corr_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["pearson"]), scale=np.std(results["pearson"]))[1]
            cca_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["cca"]), scale=np.std(results["cca"]))[1]
            print("\n{0} Iterations done\n".format(i))
            print("CCA Mean is currently {0}\nP thresh is: {1}". format(np.mean(results["cca"]),cca_p_thresh))
            
            plt.clf()
            plt.hist(results["cca"], bins=50)
            plt.axvline(cca_p_thresh, color='r', linestyle='--')    
            plt.xlabel("First Canonical Correlation")   
            plt.ylabel('Frequency')
            plt.title("Null distribution for Canonical Correlation Analysis: Radius {0}".format(radius))
            plt.show()
            #print("Correlation Mean is currently {0}\nP thresh is: {1}". format(np.mean(results["pearson"]),corr_p_thresh))
        if (i % 100 == 0):
            #corr_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["pearson"]), scale=np.std(results["pearson"]))[1]
            cca_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["cca"]), scale=np.std(results["cca"]))[1]
        
            #plt.clf()
            #plt.hist(results["pearson"], bins=50)
            #plt.axvline(corr_p_thresh, color='r', linestyle='--')    
            #plt.xlabel("Pearson's Correlation")   
            #plt.ylabel('Frequency')
            #plt.title("Null distribution for Pearson's Correlation: Radius {0}".format(radius))
            #plt.show()
            
            plt.clf()
            plt.hist(results["cca"], bins=50)
            plt.axvline(cca_p_thresh, color='r', linestyle='--')    
            plt.xlabel("First Canonical Correlation")   
            plt.ylabel('Frequency')
            plt.title("Null distribution for Canonical Correlation Analysis: Radius {0}".format(radius))
            plt.show()

    return results
    
def thresholded_isc_v_isi_analysis(cds, n=10, radius=2, alpha = 0.05, n_cpu=10):
    
    results = dict() 
    results["cca"] = []
    results["pearson"] = [] 
    corr_p_thresh = 0
    cca_p_thresh = 0
    
    for i in range(0,n):
        ds = random_shift_all_but_n(cds, i % cds.shape[0])
        cca_res = du.run_searchlight(ds, metric="1_to_many_cca", radius=radius, n_cpu=n_cpu)
        corr_res = du.run_searchlight(ds, metric="correlation", radius=radius, n_cpu=n_cpu)
        results["cca"].append(np.mean(cca_res.samples))
        results["pearson"].append(np.mean(corr_res.samples))
        
        corr_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["pearson"]), scale=np.std(results["pearson"]))[1]
        cca_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["cca"]), scale=np.std(results["cca"]))[1]
        print("\n{0} Iterations done\n".format(i))
        print("CCA Mean is currently {0}\nP thresh is: {1}". format(np.mean(results["cca"]),cca_p_thresh))
        print("Correlation Mean is currently {0}\nP thresh is{1}".format(np.mean(results["pearson"]),corr_p_thresh))
        plt.clf()
        plt.hist(results["cca"], bins=50)
        plt.axvline(cca_p_thresh, color='r', linestyle='--')    
        plt.xlabel("First Canonical Correlation")   
        plt.ylabel('Frequency')
        plt.title("Null distribution for Canonical Correlation Analysis: Radius {0}".format(radius))
        plt.show()
       
        plt.clf()
        plt.hist(results["pearson"], bins=50)
        plt.axvline(corr_p_thresh, color='r', linestyle='--')    
        plt.xlabel("Pearson's Correlation")   
        plt.ylabel('Frequency')
        plt.title("Null distribution for Pearson's Correlation: Radius {0}".format(radius))
        plt.show()
            
                      
    cca_res = du.run_searchlight(cds, metric="cca", radius=radius, n_cpu=n_cpu)
    corr_res = du.run_searchlight(cds, metric="correlation", radius=radius, n_cpu=n_cpu)
    plot_title = "ISI v. ISC: Radius {0} Thresholded For {1}% False Discovery Rate".format(radius, int(alpha * 100))
    du.plot_thresholded_isc_vs_isi(corr_res, cca_res, plot_title, corr_p_thresh,
                                   cca_p_thresh, save=True, 
                                   filename="results/figures/isc_vs_isi/thresholded_r_{0}_a_{1}.png".format(radius, int(alpha * 100)))
    return results   
    
    
    
'''
    Method to use an SVM to classify time segments between subjects. 
'''    
def between_subject_time_point_classification(ds_list=None, window=6, mask_path="../masks/aal_l_fusiform_3x3x3.nii", n_cpu=20, num_subjects=5):  
    
    if ds_list == None:
        ds_list = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu, combine=False)
    

    num_subjects = len(ds_list)
    num_samples = ds_list[0].shape[0]
    num_voxels = ds_list[0].shape[1]
    
    ds_tup = ()
    
    for ds in ds_list:
       
        ds_tup = ds_tup + ( np.repeat(ds, window, axis=0), )
        #events = find_events(targets=padded_ds.sa.targets, chunks=padded_ds.sa.chunks)
            
    cds = Dataset(np.concatenate(ds_tup))      
    cds.sa['subjects'] = np.repeat(np.arange(num_subjects), window * num_samples)
    chunked_targets = np.repeat(np.arange(num_samples), window) 
    cds.sa['targets'] = np.tile(chunked_targets, num_subjects)
    cds.sa['chunks'] = np.repeat(np.arange(num_subjects*num_samples), window)

    clf = LinearCSVMC()
       
    cv = CrossValidation(clf, NFoldPartitioner(attr='subjects'))
    cv_results = cv(cds)
    print("Mean Error: ", np.mean(cv_results))
    return cds, cv_results

    

## Making masks
## open fslview
## save a mask
## flirt -in ../LeftFrontalMedialCortex.nii.gz -ref aal_l_amyg_3x3x3.nii -out left_medial_frontal -applyxfm -usesqform
## fslchfiletype NIFTI left_medial_frontal.nii.gz

