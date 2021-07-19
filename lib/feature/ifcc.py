#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:52:13 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import librosa
import numpy as np
import scipy.fftpack
import scipy.signal
import scipy



def ditherit(x, db=-96, dith_type='db'):
    '''
    # DITHERIT Dithers a signal by db dBs.
    #   x = ditherit(x,db,dith_type)
    #
    #   Dithers a signal by db dBs
    #
    #   x:    input signal
    #   db:   dithering amount, e.g. -96
    #   dith_type: type of dithering
    #   x:    dithered signal
    #
    # ------- ditherit.m ---------------------------------------
    # Marios Athineos, marios@ee.columbia.edu
    # http://www.ee.columbia.edu/~marios/
    # Copyright (c) 2002 by Columbia University.
    # All rights reserved.
    '''    
    if dith_type=='db':
        x += (10**(db/20))*np.random.rand(np.size(x))
    elif dith_type=='bit':
#        print('ditherit: ', np.shape(x))
        x += np.round(2*db*np.random.rand(np.size(x))-db)
    return x


def linweights(L, numBands):
    step = 1/(numBands-1)
    binn = np.array(list(range(L)))/(L-1)
    W = np.zeros((L,numBands))
    for i in range(numBands):
        W[:,i] = np.exp(-0.5*np.power((binn-(i-1)*step),2)/np.power(step,2))
    midFreq = np.round(np.array(list(range(numBands)))*step*L)
    return W, midFreq



def ifccExtract(PARAMS, x, fs):
    # ncep = PARAMS['n_cep'] # earlier PARAMS['n_mfcc']=13
    x *= 2**15
    x = ditherit(x,1,'bit');
    pre_emph = False # Preemphasis is  done globally    
    if pre_emph:
        x = librosa.effects.preemphasis(x, coef=0.97)

    N = len(x)
    frameSize = int(PARAMS['Tw']*fs/1000)
    frameShift = int(PARAMS['Ts']*fs/1000)
    frames = librosa.util.frame(x, frameSize, frameShift, axis=0)
    numFrames = np.shape(frames)[0] # int(np.floor((N-frameSize)/frameShift)+1)
    N = (numFrames-1)*frameShift+frameSize
    L = int(np.floor(N/2)+1)

    x = x[:N]
    X = scipy.fftpack.fft(x)

    minFreq = 0
    maxFreq = fs/2
    minIdx = int(np.floor(2*L*minFreq/fs))
    maxIdx = int(np.floor(2*L*maxFreq/fs))
    L = maxIdx-minIdx
    # numBands = 40
    T, midFreq = linweights(L, PARAMS['numBands'])
    W = np.zeros((N, PARAMS['numBands']))
    W[minIdx:maxIdx,:] = T
    X_reap = np.repeat(np.array(X, ndmin=2).T, PARAMS['numBands'], axis=1)
    # print('WX: (%d,%d) (%d,%d) %.2fsec' % (np.shape(W)[0], np.shape(W)[1], np.shape(X_reap)[0], np.shape(X_reap)[1], len(x)/fs))
    WX = np.multiply(W, X_reap)

    midFreq = (midFreq+minIdx-1)/N*2*np.pi

    wx = scipy.fftpack.ifft(WX, n=None, axis=0)
    DWX = np.multiply(WX, np.repeat(np.array(list(range(N)), ndmin=2).T, PARAMS['numBands'], axis=1))
    dwx = scipy.fftpack.ifft(DWX, n=None, axis=0)
    instFreq = np.real(np.multiply(np.conj(wx), dwx))*2*np.pi/N

    window = [1]*(frameSize+1) # np.ones((frameSize+1,1))
    Num = np.real(np.multiply(np.conj(wx), dwx))
    Den = np.real(np.multiply(wx, np.conj(wx)))
    smif = np.zeros((N, PARAMS['numBands']))
    # print('ifcc extract: ', np.shape(window), np.shape(Num), np.shape(Den))
    for band in range(PARAMS['numBands']):
        Num[:,band] = scipy.signal.lfilter(window, 1, Num[:,band])
        Den[:,band] = scipy.signal.lfilter(window, 1, Den[:,band])
    # print('ifcc extract post filter: ', np.shape(Num), np.shape(Den))

    smif = np.divide(Num, Den+1e-10)*(2*np.pi/N)
    smif = np.subtract(smif,midFreq)
    # print('ifcc extract smif: ', np.shape(smif))

    window = np.hamming(frameSize)
    ifSpec = np.zeros((numFrames, PARAMS['numBands']))
    for i in range(numFrames):
        frameBeg = i*frameShift
        frameEnd = np.min([frameBeg + frameSize, N])
        if frameEnd-frameBeg<frameSize:
            frameBeg = frameEnd - frameSize
        smif_frame = smif[frameBeg:frameEnd,:]
        window_bands = np.repeat(np.array(window, ndmin=2).T, PARAMS['numBands'], axis=1)
        ifSpec[i, :] = np.sum(np.multiply(smif_frame, window_bands), axis=0)
    ifSpec = ifSpec[:, -1::-1]
    # print('ifcc extract ifSpec: ', np.shape(ifSpec))

    ifcc = scipy.fftpack.dct(ifSpec.T, n=PARAMS['n_cep'], axis=0).T
    # print('ifcc extract ifcc: ', np.shape(ifcc))
    
    return instFreq, ifSpec, ifcc



def compute_ifcc(PARAMS, Xin, fs):
    IF_SPEC = np.empty([])
    IFCC = np.empty([])
    # print('IFCC extract Xin_part: ', np.shape(Xin))
    
    IF, IF_SPEC, IFCC = ifccExtract(PARAMS, Xin, fs)   
    
    return IF_SPEC, IFCC
