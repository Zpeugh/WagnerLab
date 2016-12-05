# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:11:09 2016

@author: peugh.14
"""

'''====================================================================================
    Runs a correlation analysis
======================================================================================'''   
def segment_analysis(ds, t_start, t_end, metric='all', radius=2, n_cpu=20): 
   
    split_ds = Dataset(ds.samples[:, :, t_start:t_end])   
    split_ds.fa = ds.fa
    
    if metric == 'all':
        corr_res = run_searchlight(split_ds, metric='correlation', radius=radius, n_cpu=n_cpu)
        print("\nThe average correlation is: {0}".format(np.mean(corr_res.samples)))
        print("The maximum correlation is: {0}".format(np.max(corr_res.samples)))        
        
        t_res = run_searchlight(split_ds, metric='tvalues', radius=radius, n_cpu=n_cpu)       
        print("\nThe average t-value is: {0}".format(np.mean(t_res.samples)))
        print("The maximum t-value is: {0}".format(np.max(t_res.samples)))
        return corr_res, t_res
    else:
        return run_searchlight(split_ds, metric=metric, radius=radius, n_cpu=n_cpu)
   

def scene_segmentation_analysis(cds, scenes, metric='correlation', radius=2, n_cpu=20):
    scene_correlations = dict()
    scene_ds = dict()
    for i in range (0, len(scenes)-1):
        ds = segment_analysis(cds, int(scenes[i]), int(scenes[i+1]), metric=metric)
        ds_mean = np.mean(ds.samples)
        scene_correlations["scene_{0}".format(i+1)] = ds_mean
        scene_ds["scene_{0}".format(i+1)] = ds
        print("Finished scene {0}: Mean correlation was {1}".format(i, ds_mean))
        
    return scene_ds, scene_correlations



'''
    ds             The dataset
    plot_title     Should be a string with {0} in it for the run number of each plot

'''    
def plot_activation_with_scenes(ds, scenes, plot_title, window=5, a=0.01, n=34):
    
    X = ds.samples
    total_voxels = ds.shape[1]
    
    min_t = scipy.stats.t.ppf(1-a, n)    
    U = np.ma.masked_greater(X, min_t).mask  
    
    sums = np.array([np.sum(x) / float(total_voxels) for x in U])
    avgs = np.zeros_like(sums)

    plt.plot(sums)
    for i in range(0,window):
        buff = sums[:(i+1)]
        add = np.hstack((buff, sums[:-(i+1)]))
        avgs = avgs + add
    
    smoothed = avgs / float(window)

    for run in range(1,4):    
        plt.clf()    
        fig = plt.figure(figsize=(12,4))  
        run_beg = (run-1)*211        
        run_end = run*211
        plt.plot(smoothed[run_beg:run_end])
        for i, line in enumerate(scenes):
            if (line < run_end and line > run_beg):            
                plt.axvline((line - run_beg), color='r')
        plt.xlabel('Time (2.5second slices)')
        plt.ylabel("% of brain 'activated' with a={0}".format(a))
        plt.title("Run {0} Brain Activation Compared to Scene Change".format(run))
        fig.savefig(plot_title.format(run))
        plt.show()
        
        
def find_common_activation_zones_at_scene_change(cds, scenes, padding=2):

    differences = dict()
    all_scenes = []
    
    for i in range(0, len(scenes)-1):
        scene_change = scenes[i]
        voxels_before = np.mean(cds.samples[:,:,scene_change-padding:scene_change], axis=2)
        voxels_after = np.mean(cds.samples[:,:,scene_change:scene_change+padding], axis=2)
        
        t_values_before = ttest(voxels_before, popmean=0, alternative='greater')[0]
        t_values_after = ttest(voxels_after, popmean=0, alternative='greater')[0]
        
        # TODO:  make the sample shape (1, n) not (n, 1)
        samples = abs(t_values_before - t_values_after)
        samples = (samples - np.mean(samples)) / np.std(samples)
        if (i == 0):
            all_scenes = samples
        else:
            all_scenes = np.vstack((all_scenes,samples))
        print("Processed scene {0}/{1}".format(i+1, len(scenes)-1))
        ds = Dataset(samples.reshape((1, samples.shape[0])))       
        ds.fa = cds.fa
        ds.a = cds.a
        differences["scene_{0}".format(i+1)] = ds
        
    avgs = all_scenes.mean(axis=0).reshape((1, all_scenes[0].shape[0]))
    ds = Dataset(avgs)       
    ds.fa = cds.fa        
    ds.a = cds.a    
        
        
    return differences, ds
    
    
# Work in progress.
def plot_scenes(ds, a, filename=None):
    
    FALSE_DISCOVERY_RATE = 0.05
    TOTAL_VOXELS_IN_BRAIN = ds.shape[1]
    X = ds.samples
    U = np.ma.masked_inside(X, 0,a).mask
    
    sums = [np.sum(x) for x in U]    
    threshold = FALSE_DISCOVERY_RATE * TOTAL_VOXELS_IN_BRAIN
    ones = np.zeros(ds.shape[0])
    
    for i, s in enumerate(sums):
        if s > threshold:
            ones[i] = 1
            
    plt.clf()
    fig = plt.figure(figsize=(10,6))
    plt.plot(ones, '.')
    plt.xlabel('Time (2.5s increments)')
    plt.ylabel('Activated (T/F)')
    plt.ylim([0, 1.5])
    plt.title('Binary Activations')
    if filename:
       fig.savefig(filename) 
    plt.show()     