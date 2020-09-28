#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:11:27 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import librosa
import numpy as np
import scipy.fftpack



def SpectralFlux(Xin, fs, Tw, Ts):
    Nframesize = int(Tw*fs/1000)
    Nframeshift = int(Ts*fs/1000)
    frames = librosa.util.frame(Xin, Nframesize, Nframeshift, axis=0)
    nFrames = np.shape(frames)[0]
    F = np.zeros(nFrames)
    FFT = np.zeros(Nframesize)
    FFTprev = np.zeros(Nframesize)
    for i in range(nFrames):
        window = frames[i, :]
        FFT = np.abs(scipy.fftpack.fft(window, 2*Nframesize))
        FFT = FFT[:Nframesize]
        FFT /= np.max(FFT)+1e-10
        if i>0:
            F[i] = np.sum(np.power(np.subtract(FFT, FFTprev), 2))
        FFTprev = FFT
    
    return F
