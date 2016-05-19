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

cca_r2_results = du.run_searchlight(cds, nproc=58, radius=2, metric='cca')
corr_r2_results = du.run_searchlight(cds, nproc=58, radius=2, metric='correlation')

f = open('results/data/cca_34_2.pckl', 'wb')
pickle.dump(cca_r2_results, f)
f.close()

f = open('results/data/corr_34_2.pckl', 'wb')
pickle.dump(corr_r2_results, f)
f.close()

du.export_to_nifti(cca_r2_results, 'full_brain_cancorr_34_subject_r2')
du.export_to_nifti(cca_r2_results, 'full_brain_pearson_34_subject_r2')


du.plot_isc_vs_isi(corr_r2_results, cca_r2_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 2', save=True, filename='results/figures/full_brain_34_2'))


du.plot_isc_vs_isi(corr_r2_results, cca_r2_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 2', save=True, filename='results/figures/full_brain_34_2')


cca_r3_results = du.run_searchlight(cds, nproc=58, radius=3, metric='cca')
corr_r3_results = du.run_searchlight(cds, nproc=58, radius=3, metric='correlation')

f = open('results/data/cca_34_3.pckl', 'wb')
pickle.dump(cca_r3_results, f)
f.close()

f = open('results/data/corr_34_3.pckl', 'wb')
pickle.dump(corr_r3_results, f)
f.close()

du.export_to_nifti(cca_r3_results, 'full_brain_cancorr_34_subject_r3')
du.export_to_nifti(cca_r3_results, 'full_brain_pearson_34_subject_r3')

du.plot_isc_vs_isi(corr_r2_results, cca_r3_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 3', save=True, filename='results/figures/full_brain_34_3'))


du.plot_isc_vs_isi(corr_r2_results, cca_r3_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 3', save=True, filename='results/figures/full_brain_34_3')




cca_r3_results = du.run_searchlight(cds, nproc=58, radius=4, metric='cca')
corr_r4_results = du.run_searchlight(cds, nproc=58, radius=4, metric='correlation')

f = open('results/data/cca_44_4.pckl', 'wb')
pickle.dump(cca_r4_results, f)
f.close()

f = open('results/data/corr_44_4.pckl', 'wb')
pickle.dump(corr_r4_results, f)
f.close()

du.export_to_nifti(cca_r4_results, 'full_brain_cancorr_44_subject_r4')
du.export_to_nifti(cca_r4_results, 'full_brain_pearson_44_subject_r4')

du.plot_isc_vs_isi(corr_r2_results, cca_r4_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 4', save=True, filename='results/figures/full_brain_34_4'))


du.plot_isc_vs_isi(corr_r2_results, cca_r4_results, 'Full Brain 34 Subject ISI vs ISC: Seearchlight Radius 4', save=True, filename='results/figures/full_brain_34_4')


