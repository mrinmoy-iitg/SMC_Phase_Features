#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:09:39 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
import scipy.signal



def hztomel(fhz):
    '''
    THIS CONVERTS PHYSICAL FREQUENCY FREQUENCY INTO MEL FREQUENCY  FMEL=2595*(LOG10(1+FHZ/700)
    '''
    fmel = 2595*np.log10(1+fhz/700)
    return fmel



def meltohz(fmel):
    '''
    THIS CONVERTS MEL FREQUENCY INTO PHYSICAL FREQUENCY  FMEL=2595*(LOG10(1+FHZ/700)
    '''
    temp = fmel/2595
    fhz = (np.power(10, temp)-1)*700
    return fhz



def melspacing(Fmin, Fmax, N):
    '''
    # FMIN      - MINIMUM FREQUENCY (Hz)
    # FMAX      - MAXIMUM FREQUENCY (Hz)
    # N         - NO OF BANDS
    '''
    Melmin = hztomel(Fmin)
    Melmax = hztomel(Fmax)
    S = (Melmax-Melmin)/N    #SPACING
    MF = np.zeros(N+1)
    for i in range(1,N+1):
        MF[i] = MF[i-1] + S
    PF = np.zeros(N+1)
    for i in range(N+1):
        PF[i] = meltohz(MF[i])

    return MF, PF


def criticalbandfir_V3(x, F, Fs, N):
    Fs1 = Fs/2
    F1 = F[0]/(Fs1+1e-10)
    F2 = F[1]/(Fs1+1e-10)
    F3 = (F[1]+0.01*F[1])/(Fs1+1e-10)
    if F3>1:
        F2 = 0.99
        F3 = 0.999
    f = [0, F1, F2, F3,1]
    m = [0, 0, 1, 0, 0]
    # print('criticalbandfir_V3: ', F)
    # print('criticalbandfir_V3: ', f)
    b = scipy.signal.firwin2(N, f, m)
    y1 = np.convolve(b,x)
    y = y1[int(N/2):-int(N/2)]
    return y


def firfilter_lpf(x, F, Fs):
    '''    
    # F => CUTOFF FREQ
    # N => ORDER OF THE FILTER
    '''
    N = 200
    Fs1 = Fs/2
    F = F/Fs1
    b = scipy.signal.firwin(N, F)
    y1 = np.convolve(b, x, 'full')
    y = y1[int(N/2):-int(N/2)]
    return y



def segmenthamm_V2(X, FS, NW, OLN):
    NX = len(X)    # LENGTH OF THE GIVEN WAVE FILE
    NF = int(np.abs(np.floor((NX-NW+OLN)/OLN)))    # NO OF FRAMES
    samples = np.zeros((NF,NW))    # MATRIX OF DIAMENSION NO OF FRAMES x NO OF SAMPLES PER FRAME
    startingindex = np.array(list(range(NF)))*OLN + 1    # STARTING INDEX VALUE OF EACH FRAME
    w = np.hamming(NW)    # HAMMING WINDOW
    for i in range(NF):    # SEGMENTATION
        idx1 = startingindex[i]
        idx2 = np.min([startingindex[i]+NW, len(X)])
        if idx2-idx1<NW:
            idx1 = idx2-NW
        samples[i,:] = X[idx1:idx2]
        samples[i,:] = np.multiply(samples[i,:], w)
    return samples, NF, NW, OLN



def repeatvalues(x, OLN):
    y = []
    for i in range(len(x)):
        y.extend(np.repeat(x[i], OLN).tolist())
    return y


def modulationspectrum(Xin, Fs, NBANDS, NORDER, LPFFc, Fmin, Fmax, WL, OLN):
    '''
    #  TO FIND THE MODULATION FREQUENCY CONTENT (4HZ) FOR SPPECH & NON SPEECH
    #  DETECTION
    
    #Normal Values:  Fmin=0;  Fmax=4000;  NBANDS=18;  NORDER=1000; LPFFc=28; WL=20; OLN=1;  % MODULATION SPECTRUM
    
    # Xin             - SPEECH SIGNAL
    # FS              - SAMPLING FREQUENCY
    # FMIN            - MINIMUM FREQ FOR MEL FILTER
    # FMAX            - MAXIMUM FREQUENCY FOR MEL FILTER
    # NBANDS          - NO OF BANDS
    # NORDER          - BAND PASS FILTER ORDER
    # LPFFC           - LOW PASS FILTER CUTOFF FREQ
    # WL              - MODULATION SPECTRUM WINDOW LENGTH
    # OLN             - MODULATION SPECTRUM OVERLAP LENGTH
    '''
    Ampl = [];
    AMPL = [];

    #MELSPACING
    [MF,PF] = melspacing(Fmin, Fmax, NBANDS)

    #CRITICAL BAND FILTER
    ybpf = np.empty([])
    for i in range(NBANDS):
        #print('Band Pass Filter:  Band %g of %g \n' % (i,NBANDS))
        if i==0:
            F = [0.1, PF[i+1]]    #in Hz
        else:
            F = [PF[i]-0.2*PF[i], PF[i+1]]
#        ybpf[i,:] = criticalbandfir_V3(Xin, F, Fs, NORDER)
        ybpf_temp = criticalbandfir_V3(Xin, F, Fs, NORDER)
        if np.size(ybpf)<=1:
            ybpf = np.array(ybpf_temp, ndmin=2).astype(np.float32)
        else:
            ybpf = np.append(ybpf, np.array(ybpf_temp, ndmin=2).astype(np.float32), 0)
#    print('modspectrum ybpf: ', np.shape(ybpf))
    
    #HALF WAVE RECTIFICATION
    y = ybpf
    for band in range(np.shape(y)[0]):
        index = np.squeeze(np.where(y[band, :]<0))
#        print('modulationspectrum hwr: ', np.shape(y), np.shape(index), index)
        y[band, index] = 0
#    print('modspectrum y: ', np.shape(y))
    
    #ENVELOPE
    y3 = np.empty([])
    for i in range(NBANDS):
        #print('Low Pass Filter:  Band %g of %g \n' % (i,NBANDS))
        y1 = firfilter_lpf(y[i,:], LPFFc, Fs)    #LPF
        y2 = scipy.signal.decimate(y1,100, ftype='fir')    #DOWN SAMPLE
        M = np.mean(y2)    #LONG TERM AVERAGE
        y3_temp = np.divide(y2, M+1e-10)    #NORMALIZE
        if np.size(y3)<=1:
            y3 = np.array(y3_temp, ndmin=2)
        else:
            y3 = np.append(y3, np.array(y3_temp, ndmin=2), 0)
#    print('modspectrum y3: ', np.shape(y3))

    #SPECTRUM
    Mag = np.empty([])
    for i in range(NBANDS):
        #print('SPECTRUM:  Band %g of %g \n' % (i,NBANDS))
        Signal = y3[i,:]
        samples, NF, NW, OLN = segmenthamm_V2(Signal, Fs/100, WL, OLN)
        Mag_temp = np.zeros(NF)
        for j in range(NF):
            XK = np.abs(scipy.fftpack.fft(samples[j,:], 80))    #FR=80/80=1;
            Mag_temp[j] = XK[5]*XK[5]    # 4 Hz Component
        if np.size(Mag)<=1:
            Mag = np.array(Mag_temp, ndmin=2)
        else:
            Mag = np.append(Mag, np.array(Mag_temp, ndmin=2), 0)
#    print('modspectrum Mag: ', np.shape(Mag))
    
    Ampl = np.sum(Mag, axis=0)
    Ampl /= np.max(Ampl)
    AMPL1 = repeatvalues(Ampl, 100*OLN)
#    print('modulation spectrum AMPL: ', np.shape(Ampl), np.shape(AMPL1), np.shape(Xin))
    len_diff = len(Xin)-len(AMPL1)
    if len_diff>0:
        AMPL = np.multiply(np.min(AMPL1), np.ones(int(len_diff/2)))
        AMPL = np.append(AMPL, AMPL1)
        AMPL = np.append(AMPL, np.multiply(np.min(AMPL1), np.ones(len(Xin)-len(AMPL1)-(len_diff-int(len_diff/2)))))
    else:
        AMPL = AMPL1
#    print('modspectrum AMPL: ', np.shape(AMPL))

    return Ampl, AMPL
