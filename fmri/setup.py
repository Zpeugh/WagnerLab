# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:09:33 2016

@author: peugh.14
"""

from setuptools import setup

import sys, os

module_path = os.getcwd() + '/fmri'

if (sys.path.count(module_path) == 0):
    sys.path.append(os.path.abspath(os.path.join('..', 'fmri/fmri/')))

setup(name='fmri',
      version='0.1',
      description='A module to process fMRI time-series data with multivariate statistical methods',
      url='http://github.com/Zpeugh/WagnerLab',
      author='Zach Peugh',
      author_email='zachpeugh@gmail.com',
      license='MIT',
      packages=['fmri'],
      zip_safe=False)