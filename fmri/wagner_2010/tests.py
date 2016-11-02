# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:34:43 2016

@author: peugh.14
"""
import numpy as np
import analysis

def make_sin_data(samples, features, noise):
    
    X = [np.sin(x) + np.random.random() * noise - noise for x in range(samples * features)]

    return np.array(X).reshape((samples, features))


def test_svm(samples=100, features=5, noise=0.10):
    
    ds_list = []
    
    for i in range(3):
        ds_list.append(make_sin_data(samples, features, noise))
    
    cds, cv_results = analysis.between_subject_time_point_classification(ds_list=ds_list, window=4)
            

