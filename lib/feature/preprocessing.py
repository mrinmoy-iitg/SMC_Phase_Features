#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:28:27 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
import librosa



def removeSilence(Xin, fs, Tw, Ts, alpha=0.05, beta=0.1):
    Xin = Xin - np.mean(Xin)
    Xin = Xin / np.max(np.abs(Xin))
    
    frameSize = int((Tw*fs)/1000) # Frame size in number of samples
    frameShift = int((Ts*fs)/1000) # Frame shift in number of samples
    
    Rmse = librosa.feature.rms(y=Xin, frame_length=frameSize, hop_length=frameShift)
    # print('RMSE: ', np.shape(Rmse))
    energy = Rmse[0,:] #pow(Rmse,2)
    energyThresh = alpha*np.max(energy) # TEST WITH MEAN

    frame_silMarker = energy
    frame_silMarker[frame_silMarker < energyThresh] = 0
    frame_silMarker[frame_silMarker >= energyThresh] = 1
    silences = np.empty([])
    totalSilDuration = 0

    # Suppressing spurious noises -----------------------------------------        
    winSz = 20 # Giving a window size of almost 105ms for 10ms frame and 5ms shift
    i = winSz
    while i < len(frame_silMarker)-winSz:
        if np.sum(frame_silMarker[i-int(winSz/2):i+int(winSz/2)]==1) <= np.ceil(winSz*0.3):
            frame_silMarker[i] = 0
        i = i + 1
    # ---------------------------------------------------------------------
    
    sample_silMarker = np.ones(len(Xin))
    i=0
    while i<len(frame_silMarker):
        while frame_silMarker[i]==1:
            if i == len(frame_silMarker)-1:
                break
            i = i + 1
        j = i
        while frame_silMarker[j]==0:
            if j == len(frame_silMarker)-1:
                break
            j = j + 1
        k = np.max([frameShift*(i-1)+frameSize, 1])
        l = np.min([frameShift*(j-1)+frameSize,len(Xin)]);
        
        # Only silence segments of durations greater than given beta
        # (e.g. 100ms) are removed
        if (l-k)/fs > beta:
            sample_silMarker[k:l] = 0
            if np.size(silences)<=1:
                silences = np.array([k,l], ndmin=2)
            else:
                silences = np.append(silences, np.array([k,l], ndmin=2),0)
            totalSilDuration += (l-k)/fs
        i = j + 1
    
    if np.size(silences)>1:
        Xin_silrem = np.empty([]) #Xin
        for i in range(np.shape(silences)[0]):
            if i==0:
                Xin_silrem = Xin[:silences[i,0]]
            else:
                Xin_silrem = np.append(Xin_silrem, Xin[silences[i-1,1]:silences[i,0]])
    else:
        Xin_silrem = Xin
        
    return Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration




def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

