#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:41:23 2021

@author: mrinmoy
"""

import numpy as np
cimport numpy as np

np.import_array()  # needed to initialize numpy-API




def extract_patches(FV, shape, patch_size, patch_shift, labels_sp, labels_mu, training_mode):
    cdef int frmStart, frmEnd, i, nFrames, half_win
    nFrames = shape[1]
    half_win = patch_size/2
    numPatches = len(list(range(half_win, nFrames-half_win, patch_shift))) # int(np.ceil(nFrames/patch_shift))
    # print('nFrames=%d, half_win=%d, numPatches=%d' % (nFrames, half_win, numPatches))
    # print('Patches shape: (%d, %d, %d)' % (numPatches, shape[0], patch_size))

    cdef np.ndarray[double, ndim=3] patches = np.zeros((numPatches, shape[0], patch_size))
    cdef np.ndarray[double, ndim=1] patch_label_sp = np.zeros((numPatches,))
    cdef np.ndarray[double, ndim=1] patch_label_mu = np.zeros((numPatches,))
        
    nPatch = 0
    numFrame = 0
    for i in range(half_win, nFrames-half_win, patch_shift):
        frmStart = i-half_win
        frmEnd = i+half_win
        patches[nPatch,:,:] += FV[:,frmStart:frmEnd]

        if training_mode=='classification':
            ''' For classification '''
            patch_label_sp[nPatch] = labels_sp[numFrame]
            patch_label_mu[nPatch] = labels_mu[numFrame]

        elif training_mode=='regression':
            ''' For regression '''
            patch_label_sp[nPatch] = np.mean(labels_sp[frmStart:frmEnd])
            patch_label_mu[nPatch] = np.mean(labels_mu[frmStart:frmEnd])

        nPatch += 1
        numFrame += patch_shift
        # print(i, numFrame, nFrames, frmStart, frmEnd, nPatch, numPatches)
        if numFrame>=len(labels_sp):
            break
    
    return patches, patch_label_sp, patch_label_mu
