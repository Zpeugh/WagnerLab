# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:09:46 2016

@author: peugh.14
"""
from sklearn.cross_decomposition import CCA
import numpy as np
import matplotlib.pyplot as plt
from mvpa2.tutorial_suite import *
import fmri_preprocessing as fp
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from mvpa2.measures.searchlight import sphere_searchlight
from fastdtw import fastdtw
import rcca
import matplotlib.patches as mpatches
from scipy.stats import ttest_1samp as ttest

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
        ds = fp.combine_runs([ds1, ds2, ds3],correct_sr)     
        
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
        ds = fp.combine_runs([ds1, ds2, ds3],correct_sr)
        
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
    Plot the intersubject information versus intersubject canonical correlation in
    a single color scatterplot.
    
    isc_data            The Dataset object containing (1, n) intersubject canonical
                        correlation data samples
    isc_data            The Dataset object containing (1, n) intersubject correlation
                        data samples
    title               The string title for the figure
    save                (optional) Boolean variable for whether or not you wish to save
                        the file
    filename            Required if save=True. The name of the file to save the figure as

    voxel_position      a number representing which voxel in the dataset to display
======================================================================================'''  
def plot_isc_vs_isi(isc_data, isi_data, title, save=False, filename=None):
    plt.clf()    
    fig = plt.figure(figsize=(10,6))
    plt.scatter(isc_data, isi_data)
    plt.title(title)
    plt.xlabel('Intersubject Correlation')
    plt.ylabel('intersubject Information')
    
    if save:
        fig.savefig(filename)
    plt.show()
 

'''====================================================================================
    Plot the intersubject information versus intersubject canonical correlation in
    a tri-colored scatterplot, where blue indicates the back 1/3rd of the brain, green
    indicates the middle 1/3rd and red indicates the front 1/3rd (all with respect to the
    sagittal plane).
    
    isc_data            The Dataset object containing (1, n) intersubject canonical
                        correlation data samples
    isc_data            The Dataset object containing (1, n) intersubject correlation
                        data samples
    title               The string title for the figure
    save                (optional) Boolean variable for whether or not you wish to save
                        the file
    filename            Required if save=True. The name of the file to save the figure as

    voxel_position      a number representing which voxel in the dataset to display
======================================================================================'''  
def plot_colored_isc_vs_isi(isc_data, isi_data, title, save=False, filename=None):
   
    voxels = isc_data.fa.voxel_indices    
    
    iscd = isc_data.samples[0,:]
    isid = isi_data.samples[0,:]
    
    blue_x = [x for i, x in enumerate(iscd) if voxels[i][1] <= 20]
    blue_y = [x for i, x in enumerate(isid) if voxels[i][1] <= 20]
    
    green_x = [x for i, x in enumerate(iscd) if voxels[i][1] > 20 and voxels[i][1] < 41]
    green_y = [x for i, x in enumerate(isid) if voxels[i][1] > 20 and voxels[i][1] < 41]
    
    red_x = [x for i, x in enumerate(iscd) if voxels[i][1] >= 41]
    red_y = [x for i, x in enumerate(isid) if voxels[i][1] >= 41]
    
    color_array = ['seagreen' for i in range(len(green_x))]
    label_array = ['Middle brain' for i in range(len(green_x))]
    color_array += ['darkred' for i in range(len(red_x))] 
    label_array = ['Front brain' for i in range(len(red_x))]
    color_array += ['steelblue' for i in range(len(blue_x))]       
    label_array = ['Back brain' for i in range(len(blue_x))] 
    
    X = green_x + red_x + blue_x
    Y = green_y + red_y + blue_y

    plt.clf()
    fig = plt.figure(figsize=(10,6))
    blue_marker = mpatches.Patch(color='steelblue', label="back")
    green_marker = mpatches.Patch(color='seagreen', label="middle")
    red_marker = mpatches.Patch(color='darkred', label="front")
    
    plt.scatter(X,Y, marker = 'x', color=color_array)
    fig.legend(handles=[blue_marker, green_marker, red_marker], labels=["Back 1/3rd", "Middle 1/3rd", "Front 1/3rd"], bbox_to_anchor=(0.66,0.125), loc='lower left')
    plt.title(title, fontsize=15)
    plt.xlabel('Intersubject Correlation',fontsize=15)
    plt.ylabel('Intersubject Information', fontsize=15)
    
    if save:
        fig.savefig(filename)
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
    
 
  
'''=======================================================================================
    Given a dataset with shape (s, v, t) where 
        s is the number of subjects
        v is the number of voxels in each subject
        t is the constant number of time series for each voxel
    This function will average across the voxels and then compute all combinations
    of pairwise pearsons correlations between subjects s.  The mean of this is returned.
======================================================================================='''
def pearsons_average(ds):
  
    return 1 - np.mean( pdist(np.mean(ds.samples, axis=1), metric='correlation') )



'''=======================================================================================
    Given a dataset with shape (s, v, t) where 
        s is the number of subjects
        v is the number of voxels in each subject
        t is the constant number of time series for each voxel
    All voxels in the searchlight are mean centered at each time point.  All combinations 
    of pairwise 1st canonical correlations are calculated and the mean value for this
    region is returned.
======================================================================================='''    
def cca(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    cca.train([subj.T for subj in centered_ds]) 
    return np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
  


def cca_validate(ds):
    num_subj = ds.shape[0]
    num_samples = ds.shape[2]
    #split_point = int(num_samples * .8)
    split_point = int(num_subj * .66)
    
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    centered_ds = ds.samples - np.mean(np.mean(ds.samples, axis=1), axis=0)
    
    
    #train_set = [subj.T[:split_point,:] for subj in centered_ds]
    #test_set = [subj.T[split_point:,:] for subj in centered_ds]
    
    train_set = [subj.T for subj in centered_ds[:split_point,:,:]]
    test_set = [subj.T for subj in centered_ds[split_point:,:,:]]
    
    cca.train(train_set)
    cca.validate(test_set)
    
    cancorr = np.mean(cca.cancorrs[0][np.triu_indices(split_point,k=1)])
    predcorr = np.max(cca.corrs)
    return np.array([cancorr, predcorr])

def first_canonical_correlation(u, v):
    
    cca = CCA(n_components=1)    
    U_c, V_c = cca.fit_transform(u, v) 
    return np.corrcoef(U_c.T, V_c.T)[0,1]


def sk_cca(ds):
    num_subj = ds.shape[0]
    corrs= []
    for i in range(0, num_subj-1):
        for j in range(i, num_subj):            
            U = ds.samples[i,:,:].T
            V = ds.samples[j, :, :].T            
            corrs.append(first_canonical_correlation(U, V))
    return np.mean(corrs)

'''=======================================================================================
    Given a dataset with shape (s, v, t) where 
        s is the number of subjects
        v is the number of voxels in each subject
        t is the constant number of time series for each voxel
    This function will compute all combinations of pairwise 1st canonical correlation
    coefficients between subjects s.  The mean of these is returned.
======================================================================================='''    
def cca_uncentered(ds):
    num_subj = ds.shape[0]
    cca = rcca.CCA(kernelcca=False, numCC=1, reg=0., verbose=False)
    cca.train([subj.T for subj in ds.samples]) 
    return np.mean(cca.cancorrs[0][np.triu_indices(num_subj,k=1)])
    
    
'''=======================================================================================
    Given a dataset with shape (s, v, t) where 
        s is the number of subjects
        v is the number of voxels in each subject
        t is the constant number of time series for each voxel
    This function will average across the voxels and then compute all combinations
    of pairwise euclidean distances between subjects s.  The mean of this is returned.
======================================================================================='''
def euclidean_average(ds):  
    return np.mean(pdist(np.mean(ds.samples, axis=1), metric='euclidean'))




def pvalues(ds):   
    return ttest(ds.samples.mean(axis=1), 0).pvalue

'''=======================================================================================
    Given a dataset with shape (s, v, t) where 
        s is the number of subjects
        v is the number of voxels in each subject
        t is the constant number of time series for each voxel
    This function will average across the voxels and then compute all combinations
    of pairwise dynamic time warped distances between subjects s. The mean of this 
    is returned.
======================================================================================='''
def dtw_average(ds):	
    X = np.mean(ds.samples, axis=1)
    return np.mean(pdist(X, lambda u, v: fastdtw(u, v)[0]))



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
    ds_tup = ()
    for subj in dslist:
        ds_tup = ds_tup + (subj.samples.T,)        
    ds = Dataset(np.vstack(ds_tup).reshape((len(dslist), num_voxels, num_samples)))
    ds.a.mapper = dslist[0].a.mapper
    ds.fa["voxel_indices"] = dslist[0].fa.voxel_indices
    ds.sa.clear()
    ds.sa["subject"] = np.arange(len(dslist))       
    return ds

    
'''====================================================================================
    Takes a dataset of shape (n, v, t) where n is number of subjects, v is number
    of voxels, and t is number of time samples for each subject.  Runs a parallel
    searchlight analysis on all of the subjects given the metric input.
    
    ds          The Dataset object containing all of the subjects runs
    metric      (optional) One of 'euclidean', 'dtw', 'cca', 'correlation', 'pvalues'.
                Defaults to Pearsons Correlation. 
    radius      (optional) The radius of the searchlight sphere. Defaults to 3.
    center_ids  (optional) The feature attribute name to use as the centers for the 
                searchlights.
    nproc       (optional) Number of processors to use.  Defaults to all available 
                processors on the system. 
                        
    Returns      The results of the searchlight
======================================================================================''' 
def run_searchlight(ds, metric='correlation', radius=3, center_ids=None, n_cpu=40):

    if metric == 'euclidean':
        measure = euclidean_average
    elif metric == 'dtw':
        measure = dtw_average
    elif metric == 'cca_u':
        measure = cca_uncentered
    elif metric == 'cca':
        measure = cca
    elif metric == 'sk_cca':
        measure = seq_cca
    elif metric == 'cca_validate':
        measure = cca_validate
    elif metric == 'correlation':
        measure = pearsons_average
    elif metric == 'pvalues':
        measure = pvalues
    else:
        print("Invalid metric, using Pearson's Correlation by default.")
        measure = pearsons_average
        
    sl = sphere_searchlight(measure, radius=radius, center_ids=center_ids, nproc=n_cpu)   
    
    searched_ds = sl(ds)
    searched_ds.fa = ds.fa
    searched_ds.a = ds.a    
    
    return searched_ds
   
   
   
   
   
#########################################################################################  



'''====================================================================================
    This should take the mean accross the searchlight of voxels, then look through 
    all n voxels at time t, and return argmax(), i.e the voxel coordinate of the 
    'most activated' location for that time.
======================================================================================''' 
def maximum_time():
    return 1        

  
# takes a sphere through a subject with boolean activation s and returns the number
# of voxels in that where reported as active for each time series. Returns an array
# of shape (n, t) where n is the number of active voxels and t is the fixed number of
# samples over time.
def count_active(ds):
    #return 1
    return np.sum(ds.samples.mask, axis=1)
    
    
def get_most_active_indices(ds):
    k = 1    
    #return np.argsort(ds)[:,-k:]
    return np.argmax(ds, axis=1)    
    

#Takes a dataset and searches at each time point for the k most active regions
#def find_active_regions(ds):
#    data = ds.samples
#    voxels = []
#    for t_slice in data:
#        voxels.append(ds.fa.voxel_indices[np.argmax(t_slice)])                    
#    return voxels

# takes a dataset and returns the coordinates of its 5 most activated regions per time
# point.  The shape of the resulting array is (t, 5, 3) where t is number of time series
def find_active_regions(dslist, sd_threshold=3.0):   
    
    ds_copy = ds.copy()
        
    ds_copy.samples = np.ma.masked_greater(ds.samples, sd_threshold)
    sl = sphere_searchlight(count_active, radius=3, nproc=58)
    
    searched_ds = sl(ds_copy)
    return searched_ds
#    results = []
#    voxel_indices = ds.fa.voxel_indices
#    for row in searched_ds:
#        top_five = []
#        for entry in row:
#            top_five.append(voxel_indices[entry])
#        results.append(top_five)
#    return np.array(results)


## TODO:  First run all 34 subjects through a searchlight of radius r, averaging
## TODO:  the voxel space and putting the means back in the voxel centers, to adjust
## TODO:  for artifact detection, anatomical difference, and motion.  Then,  
## TODO:  find out a way to use Searchlight and create a 3D voxel_space which is
## TODO:  just linear. Then only allow a radius of 1, so the parallelization happens
## TODO:  across 633 time points.  The distance measure should take in a dataset of
## TODO:  shape (34, 1, 96068), and return the voxel index [x, y, z] of the maximum
## TODO:  average searchlight.  Or you could write your own shit and parallelize it.

    
#==============================================================================
#     
# def run_spatio_temporal_searchlight(ds, metric='correlation', s_rad=2, t_rad=2, nproc=58):
#     
#     measure = pearsons_average
#    
#     if metric == 'euclidean':
#         measure = euclidean_average
#     if metric == 'dtw':
#         measure = dtw_average
#         
#     sl = Searchlight(measure, IndexQueryEngine( voxel_indices = Sphere(radius),
#                                                 time_attr = Sphere(r_rad) ),
#                                                 nroc = nproc )
#     
#     searched_ds = sl(ds)
#     searched_ds.fa = ds.fa
#     searched_ds.a = ds.a    
#     
#     return searched_ds
#     
#==============================================================================

