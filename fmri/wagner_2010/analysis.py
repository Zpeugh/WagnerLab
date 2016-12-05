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
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS as MDS
from scipy.cluster import hierarchy 

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
        corr_res = du.run_searchlight(ds, metric="correlation", radius=radius, n_cpu=n_cpu)
        results["cca"].append(np.mean(cca_res.samples))
        results["pearson"].append(np.mean(corr_res.samples))
        if (i % 2 == 0):
            corr_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["pearson"]), scale=np.std(results["pearson"]))[1]
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
            print("Correlation Mean is currently {0}\nP thresh is: {1}". format(np.mean(results["pearson"]),corr_p_thresh))
        if (i % 100 == 0):
            corr_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["pearson"]), scale=np.std(results["pearson"]))[1]
            cca_p_thresh = stats.norm.interval(1-alpha, loc=np.mean(results["cca"]), scale=np.std(results["cca"]))[1]
        
            plt.clf()
            plt.hist(results["pearson"], bins=50)
            plt.axvline(corr_p_thresh, color='r', linestyle='--')    
            plt.xlabel("Pearson's Correlation")   
            plt.ylabel('Frequency')
            plt.title("Null distribution for Pearson's Correlation: Radius {0}".format(radius))
            plt.show()
            
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
    
    
    
def scene_svm_cross_validation(cds=None, mask_path="../masks/aal_l_hippocampus_3x3x3.nii", n_cpu=20, num_subjects=5):  
    
    if cds == None:
        cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu)
    

    cds.a["scene_changes"] = ld.get_2010_scene_splits(as_ints=True)
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    num_scenes = len(cds.a.scene_changes)
    ds_list = np.zeros((num_subj, num_voxels, num_scenes))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i, scene_cutoff in enumerate(cds.a.scene_changes):
        ds_list[:,:,i] = np.mean(cds.samples[:,:,prev_cutoff:scene_cutoff], axis=2)
        prev_cutoff = scene_cutoff
       
    for subj in ds_list:
        ds_tup = ds_tup + (subj.T, )
        
        
    ds = Dataset(np.concatenate(ds_tup))  
    print(ds.shape)
    ds.sa['subjects'] = np.repeat(np.arange(num_subj), num_scenes)
    ds.sa['targets'] = np.tile(np.arange(num_scenes), num_subj)
    ds.sa['chunks'] = np.tile(np.arange(num_scenes), num_subj)

    clf = SVM()
       
    cv = CrossValidation(clf, NFoldPartitioner(attr='subjects'))
    cv_results = cv(ds)
    print("Mean Accuracy: ", 1 - np.mean(cv_results))
    return cds, cv_results

    

    

def scene_double_correlation(mask_path="../masks/bigmask_3x3x3.nii", file_prefix="full_brain", 
                             n_cpu=40, num_subjects=34, radii = [2,3,4,5]):
    
    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu)
    
    cds.a["scene_changes"] = ld.get_2010_scene_splits(as_ints=True)
    results = []
    
    for radius in radii:
        res = du.run_searchlight(cds, metric="scene_based_double_corr", radius=radius, n_cpu=n_cpu)
        du.export_to_nifti(res, "results/nifti/scene_splits/{0}_s{1}_r{2}".format(file_prefix, num_subjects, radius))
        results.append(res)
        
    return results
    
    
def export_scenes_to_nifti(ds, roi, radius):
    i = 1
    if not os.path.exists("results/nifti/scenes/{0}".format(roi)):
        os.mkdir("results/nifti/scenes/{0}".format(roi))
    if not os.path.exists("results/nifti/scenes/{0}/radius_{1}".format(roi, radius)):
        os.mkdir("results/nifti/scenes/{0}/radius_{1}".format(roi, radius))
    
    for scene in ds.samples.T:
        temp_ds = Dataset(scene.reshape((1, len(scene))))
        temp_ds.fa = ds.fa
        temp_ds.a = ds.a
        du.export_to_nifti(temp_ds, "results/nifti/scenes/{0}/radius_{1}/scene_{2}".format(roi, radius, i))
        i += 1
    
    
def scene_svm_cross_validation_conf_mat(mask_path="../masks/bigmask_3x3x3.nii", roi="full_brain", 
                             n_cpu=40, num_subjects=34, radii = [2,3,4], scenes=None):
    
    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu)
    
    if scenes == None:
        cds.a["scene_changes"] = ld.get_2010_scene_splits(as_ints=True)
    else:
        cds.a["scene_changes"] = scenes
        
    results = dict()
    
    for radius in radii:
        ds = du.run_searchlight(cds, metric="scene_svm_cv_cm", radius=radius, n_cpu=n_cpu)
        num_voxels = ds.shape[1]
        num_scenes = len(cds.a.scene_changes)
        accuracies = du.confusion_matrix_accuracies(ds.samples.T.reshape((num_voxels,num_scenes,num_scenes)))
        res = Dataset(np.array(accuracies))
        res.a = cds.a
        res.fa = cds.fa
        res.sa = cds.sa
        export_scenes_to_nifti(res, roi, radius)
        results["radius_{0}".format(radius)] = res
        print("Finished radius {0}".format(radius))

                
    return results
    

    
def temp_script():
    results = scene_svm_cross_validation_conf_mat(n_cpu=45, num_subjects=34, radii = [2,3,4])
    print("Done with scene wise splitting")
    scenes = [j for i in zip(np.arange(0,633-6,6),np.arange(6,633,6)) for j in i]
    results2 = scene_svm_cross_validation_conf_mat(n_cpu=45, num_subjects=34, radii = [2,3,4], roi="full_brain_window_6", scenes=scenes)
    return (results, results2)



def scene_based_mds(num_subjects=34, mask_path="../masks/aal_l_fusiform_3x3x3.nii", n_cpu=34):    
    
    cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu)
    cds.a["scene_changes"] = ld.get_2010_scene_splits()


    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    scenes = cds.a.scene_changes
    num_scenes = len(scenes)
    ds_list = np.zeros((num_subj, num_voxels, num_scenes-1))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_scenes - 1):
        ds_list[:,:,i] = np.mean(cds.samples[:,:,scenes[i]:scenes[i+1]], axis=2)
       
    dsm_array = []    
    for subj in ds_list:
        
        dsm_array.append(squareform(1 - pdist(subj.T, metric='correlation')))
        
    dsm = np.mean(dsm_array, axis=0)
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    coords = mds.fit(dsm).embedding_
    
    plt.clf()
    X, Y = coords[:,0], coords[:,1]
    labels = np.arange(1,num_scenes)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    plt.scatter(X,Y, marker='x')
    for i, label in enumerate(np.arange(1,num_scenes)):
        ax.annotate(label, (X[i],Y[i]))    
        
    plt.axis([np.min(X)*1.2, np.max(X)*1.2, np.min(Y)*1.2, np.max(Y)*1.2])
    plt.title("MDS Scene Visualization")
    plt.show()
    
    return dsm

def show_dendrogram(cds=None, scenes=None, num_subjects=34, mask_path="../masks/aal_l_fusiform_3x3x3.nii", n_cpu=34):    
    
    if cds is None:      
        cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu)
    if scenes is not None:
        cds.a["scene_changes"] = scenes
    else:        
        cds.a["scene_changes"] = ld.get_2010_scene_splits()


    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    scenes = cds.a.scene_changes
    num_scenes = len(scenes)
    ds_list = np.zeros((num_subj, num_voxels, num_scenes-1))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_scenes - 1):
        ds_list[:,:,i] = np.mean(cds.samples[:,:,scenes[i]:scenes[i+1]], axis=2)
       
    Z = hierarchy.linkage(np.mean(ds_list, axis=0).T, metric='correlation')
        
    plt.figure(figsize=(14,8))
    hierarchy.dendrogram(Z)
    plt.show()
    return Z
    
def principal_voxel_svm(cds=None, scenes=None, mask_path="../masks/aal_l_fusiform_3x3x3.nii", n_cpu=20, num_subjects=5):  
    
    if cds == None:
        cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu, combine=False)
    if scenes is not None:
        cds.a["scene_changes"] = scenes
    else:        
        cds.a["scene_changes"] = ld.get_2010_scene_splits()
    
    num_subj = cds.shape[0]
    num_voxels = cds.shape[1]
    scenes = cds.a.scene_changes
    num_scenes = len(scenes) - 1
    ds_list = np.zeros((num_subj, num_voxels, num_scenes))
    prev_cutoff = 0
    ds_tup = ()
    
    # average correlations for each scene
    for i in range(num_scenes - 1):
        ds_list[:,:,i] = np.mean(cds.samples[:,:,scenes[i]:scenes[i+1]], axis=2)
    
    for ds in ds_list: 
        ds_tup = ds_tup + ( np.repeat(ds, window, axis=0), )
            
    cds = Dataset(np.concatenate(ds_tup))      
    cds.sa['subjects'] = np.repeat(np.arange(num_subjects), num_scenes * num_voxels)
    chunked_targets = np.repeat(np.arange(num_samples), num_scenes) 
    cds.sa['targets'] = np.tile(chunked_targets, num_subjects)
    cds.sa['chunks'] = np.repeat(np.arange(num_subjects*num_samples), num_scenes)

    clf = SVM()
       
    cv = CrossValidation(clf, NFoldPartitioner(attr='subjects'))
    cv_results = cv(cds)
    print("Mean Error: ", np.mean(cv_results))
    return cds, cv_results
    
    
    
def get_average_scene_bounds(cds=None, scenes=None, mask_path="../masks/aal_l_fusiform_3x3x3.nii", n_cpu=34, num_subjects=34, radius=2):  
    
    if cds == None:
        cds = ld.get_2010_preprocessed_data(num_subjects=num_subjects, mask_path=mask_path, n_cpu=n_cpu, combine=True)
    if scenes is not None:
        cds.a["scene_changes"] = scenes
    else:        
        cds.a["scene_changes"] = ld.get_2010_scene_splits()
        
    cds.a["clusters_per_iter"] = 80
    ds = du.run_searchlight(cds, metric='cluster_scenes_return_indices', n_cpu=n_cpu, radius=radius)
    
    return ds    
    
    
## Making masks
## open fslview
## save a mask
## flirt -in ../LeftFrontalMedialCortex.nii.gz -ref aal_l_amyg_3x3x3.nii -out left_medial_frontal -applyxfm -usesqform
## fslchfiletype NIFTI left_medial_frontal.nii.gz

