#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:49:36 2021

@author: mrinmoy
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# setup(
#       ext_modules=[
#           Extension(
#               'cython_funcs', 
#               ['cython_funcs.c'], 
#               include_dirs=[numpy.get_include()]
#               )
#           ,],)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
      ext_modules=cythonize('cython_funcs.pyx'), 
      include_dirs=[numpy.get_include()])    


# Running the script: python setup.py build_ext --inplace
