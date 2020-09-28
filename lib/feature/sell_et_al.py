#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:34:08 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import librosa
import numpy as np
import lib.feature.spectral_flux as spflux
import lib.feature.modulation_spectrum as modspec
import scipy.fftpack





def computeChromaFeatures(Xin, fs, Tw, Ts, K):
    '''
    # Function to compute the chromagram of audio and features from that.
    '''
    nPts = int(Tw*fs/1000)
    nShft = int(Ts*fs/1000)
    Chromagram = librosa.feature.chroma_stft(y=Xin, sr=fs, n_fft=nPts, hop_length=nShft, center=False, n_chroma=K)
    lmin = 6
    lmax = 18
    
    nFrames = np.shape(Chromagram)[1]
    
    # Chroma Difference feature
    chromaDiff = np.zeros(nFrames)
    for i in range(nFrames):
        c = Chromagram[:,i]
        cdf = 0
        for j in range(K):
            cdf += np.power(np.abs(c[j] - c[np.max([(j+1)%K,1])]), 2)
        chromaDiff[i] = cdf
        
    # Chroma High Frequency feature
    chromaHighFreq = np.zeros(nFrames)
    for i in range(nFrames):
        c = Chromagram[:,i]
        chf = 0
        Cf = np.abs(scipy.fftpack.fft(c,2*K))
        Cf = Cf[:K]
        for j in range(lmin, lmax):
            chf += np.power(Cf[j], 2)
        chromaHighFreq[i] = chf
        
    return Chromagram, chromaDiff, chromaHighFreq





def computeSilentIntervalFeatures(SilInt, nFrames):
    '''
    # Input's are speech signal in column vector form,sampling frequency in Hz,
    # frame size and frame shift in samples, total no. of frames in the signal
    #Output:
    #   silRatio = Silent Interval Ratio
    #   silFrequency = Silent Interval Frequency
    '''
    silRatio = []
    silFrequency = []
    silCount = 0
    audibleCount = 0
    i = 0
    while i<len(SilInt):
        additionAud = 0
        while SilInt[i]==1:
            i += 1
            additionAud = 1
            if i==len(SilInt):
                break
        audibleCount += additionAud
        if i==len(SilInt):
            continue
        additionSil = 0
        while SilInt[i]==0:
            i += 1
            additionSil = 1
            if i==len(SilInt):
                break
        silCount += additionSil
        
    silRatio = np.sum(SilInt==0)/(np.sum(SilInt==0)+np.sum(SilInt==1))
    silFrequency = silCount/(silCount+audibleCount+1e-10)
    SR = np.repeat(silRatio, nFrames)
    SF = np.repeat(silFrequency, nFrames)

    return SR, SF




def compute_Sell_et_al_features(PARAMS, Xin, fs, frame_silMarker):
    Nframesize = int(PARAMS['Tw']*fs/1000)
    Nframeshift = int(PARAMS['Ts']*fs/1000)
    FV = np.empty([])

    Rms = librosa.feature.rms(y=Xin, frame_length=Nframesize, hop_length=Nframeshift, center=False)
    # print('\tRms: ', np.shape(Rms))
    FV = np.array(Rms, ndmin=2).T

    ZCR = librosa.feature.zero_crossing_rate(y=Xin, frame_length=Nframesize, hop_length=Nframeshift, center=False)
    # print('\tZCR: ', np.shape(ZCR))
    FV = np.append(FV, np.array(ZCR, ndmin=2).T, axis=1)

    Spectral_Centroid = librosa.feature.spectral_centroid(y=Xin, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, center=False)
    # print('\tSpectral_Centroid: ', np.shape(Spectral_Centroid))
    FV = np.append(FV, np.array(Spectral_Centroid, ndmin=2).T, axis=1)
    
    Spectral_Flux = spflux.SpectralFlux(Xin, fs, PARAMS['Tw'], PARAMS['Ts'])
    # print('\tSpectral_Flux: ', np.shape(Spectral_Flux))
    FV = np.append(FV, np.array(Spectral_Flux, ndmin=2).T, axis=1)
    
    silRatio, silFrequency = computeSilentIntervalFeatures(frame_silMarker, np.shape(FV)[0])
    # print('\tSilRatio: ', np.shape(silRatio), np.shape(silFrequency))
    FV = np.append(FV, np.array(silRatio, ndmin=2).T, axis=1)
    FV = np.append(FV, np.array(silFrequency, ndmin=2).T, axis=1)

    ms1, ModSpec = modspec.modulationspectrum(Xin, fs, PARAMS['NBANDS'], PARAMS['NORDER'], PARAMS['LPFFc'], Fmin=0, Fmax=fs/2, WL=PARAMS['Tw'], OLN=PARAMS['Ts'])
    ModSpec_frames = librosa.util.frame(ModSpec, Nframesize, Nframeshift, axis=0)
    ModSpec_feat = np.mean(ModSpec_frames, axis=1)
    # print('\tModSpec_feat: ', np.shape(ModSpec_feat))
    FV = np.append(FV, np.array(ModSpec_feat, ndmin=2).T, axis=1)

    Chromagram, chromaDiff, chromaHighFreq = computeChromaFeatures(Xin, fs, PARAMS['Tw'], PARAMS['Ts'], PARAMS['K']);
    # print('\tChromagram: ', np.shape(chromaDiff), np.shape(chromaHighFreq))
    FV = np.append(FV, np.array(chromaDiff, ndmin=2).T, axis=1)
    FV = np.append(FV, np.array(chromaHighFreq, ndmin=2).T, axis=1)

    return FV.astype(np.float32)