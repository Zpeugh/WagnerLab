# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:23:58 2016

@author: peugh.14
"""

import multithreaded as mult
import dataset_utilities as du
import pickle

dslist = mult.get_2010_preprocessed_data()
cds = du.combine_datasets(dslist)

################################Radius 3#########################################
cca_r3_results = du.run_searchlight(cds, n_cpu=40, radius=3, metric='cca')

f = open('results/data/cca_34_3.pckl', 'wb')
pickle.dump(cca_r3_results, f)
f.close()

du.export_to_nifti(cca_r3_results, 'full_brain_cancorr_34_subject_r3')


corr_r3_results = du.run_searchlight(cds, n_cpu=40, radius=3, metric='correlation')

f = open('results/data/corr_34_3.pckl', 'wb')
pickle.dump(corr_r3_results, f)
f.close()

du.export_to_nifti(corr_r3_results, 'full_brain_pearson_34_subject_r3')


du.plot_isc_vs_isi(corr_r3_results, cca_r3_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 3', save=True, filename='results/figures/full_brain_34_3')


###################################Radius 4#####################################
cca_r4_results = du.run_searchlight(cds, n_cpu=40, radius=4, metric='cca')

f = open('results/data/cca_34_4.pckl', 'wb')
pickle.dump(cca_r4_results, f)
f.close()

du.export_to_nifti(cca_r4_results, 'full_brain_cancorr_34_subject_r4')

corr_r4_results = du.run_searchlight(cds, n_cpu=40, radius=4, metric='correlation')

f = open('results/data/corr_34_4.pckl', 'wb')
pickle.dump(corr_r4_results, f)
f.close()

du.export_to_nifti(corr_r4_results, 'full_brain_pearson_34_subject_r4')


du.plot_isc_vs_isi(corr_r4_results, cca_r4_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 4', save=True, filename='results/figures/full_brain_34_4')




