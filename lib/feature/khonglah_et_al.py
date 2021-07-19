#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:31:11 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import librosa
import scipy.signal
import scipy.fftpack
import lib.feature.modulation_spectrum as modspec
import lib.feature.spectral_flux as spflux



def cumsum_new(x):
    '''
    The numpy cumsum function has some issues. It adds some step-ladder type 
    high-frequency information in the output signal. When removing trend from this
    output, the step-ladder information keep increasing in amplitude and show up
    as the trend-removed signal's envelope. The trend does get removed. Therefore
    this function has been written to get the expected cumsum output.
    '''
    y = np.zeros(np.size(x))
    y[0] = x[0]
    for i in range(1, np.size(x)):
        y[i] = x[i] + y[i-1]
    return y




def remTrend(sig, winSize):
    sig = sig.flatten()
    window = [1]*winSize
    rm = np.convolve(sig, window, mode='same')
    #rm = rm[int(winSize/2)-1:-int(winSize/2)]
    norm = np.convolve([1]*len(sig), window, mode='same')
    #norm = norm[int(winSize/2)-1:-int(winSize/2)]
    rm_norm = np.divide(rm, norm)
    out = sig-rm_norm
    return out



def zeroFreqFilter(wav, fs, winLength):
    dwav = np.diff(wav)
    dwav = np.append(dwav, dwav[-1])
    dwav = dwav/(np.max(np.abs(dwav)))
#    print('zeroFreqFilter dwav: ', np.shape(wav), np.shape(dwav), np.max(np.abs(dwav)))
    N = len(dwav)
    zfSig = cumsum_new(cumsum_new(cumsum_new(cumsum_new(dwav))))
    winLength = int(np.round(winLength*fs/1000))
#    print('zeroFreqFilter winLength: ', winLength)
    zfSig = remTrend(zfSig, winLength)
    zfSig = remTrend(zfSig, winLength)
    zfSig = remTrend(zfSig, winLength)
    zfSig = remTrend(zfSig, winLength)
    zfSig[N-winLength*2:] = 0
    zfSig[:winLength*2] = 0
    return zfSig


def blockproc_func(x):
    num = np.multiply((x-np.mean(x)), np.hamming(len(x)))
    num = np.correlate(num, num, mode='same')
    den = np.hamming(len(x))
    den = np.correlate(den, den, mode='same')
    return np.divide(num, den+1e-10)


def blockproc(X, block_shape, func):
    X_flatten = X.flatten()
    X_size = np.size(X_flatten)
    block_size = block_shape[0]*block_shape[1]
    if X_size%block_size>0:
        X_flatten = np.append(X_flatten, np.zeros(X_size%block_size))
    X_size = np.size(X_flatten)
    numBlocks = int(np.ceil(np.size(X_flatten)/block_size))
    block_proc_out = np.empty([])
    for i in range(numBlocks):
        func_out = func(X_flatten[i*block_size:(i+1)*block_size])
        if np.size(block_proc_out)<=1:
            block_proc_out = np.array(func_out, ndmin=2).T
        else:
            block_proc_out = np.append(block_proc_out, np.array(func_out, ndmin=2).T, 1)
    return block_proc_out


def xcorrWinLen(PARAMS, wav, fs):
    zfSig = zeroFreqFilter(wav, fs, 2)
    zfSig = zfSig/(np.max(np.abs(zfSig))+1e-10)
    wav = zfSig
    
    frameSize = int(PARAMS['Tw']*fs/1000)
    frameShift = int(PARAMS['Ts']*fs/1000)
    
    en = np.convolve(np.power(wav, 2), [1]*frameSize)
    en = en[int(frameSize/2):-1-int(frameSize/2)]
    en = en/frameSize
    en = np.abs(np.sqrt(en))
    en = np.array(en>np.max(en)).astype(int)/5
    
    b = librosa.util.frame(wav, frameSize, frameShift, axis=0).T
    vad = librosa.util.frame(en, frameSize, frameShift, axis=0).T
    blockproc_out = blockproc(b, [frameSize,1], blockproc_func)
    blockproc_out = blockproc_out[frameSize:,:]
    
    minPitch = 3    #2 ms == 500 Hz.
    maxPitch = 16    #16 ms == 66.66 Hz.
    #maxv = np.max(out[int(minPitch*fs/1000): int(maxPitch*fs/1000), :], axis=0)
    maxi = np.argmax(blockproc_out[int(minPitch*fs/1000): int(maxPitch*fs/1000), :], axis=0)
    x = (np.arange(minPitch, maxPitch, 0.5)*fs/1000)+2
    pLoc = maxi[np.squeeze(np.where(vad>(frameSize*0.8)))]+minPitch*fs/1000
    y = np.histogram(pLoc, x)
    y = np.divide(y, len(pLoc)+1e-10)
    
    #val = np.max(y)
    idx = np.argmax(y)
    print('xcorrWinLen idx: ', np.shape(idx), np.shape(y))
    idx = int(np.round(idx/2)+minPitch+1)
    print('Average pitch period: ', idx, ' ms')
    
    return idx



def zfsig2(PARAMS, wav, fs, winLength):
    '''
    # function [zf,gci,es,f0] = zfsig(wav,fs,winLength)
    # 
    # Returns:
    #       zf      - zero-frequency resonator signal
    #       gci     - glottal closure instants
    #       es      - excitation strength at GCIs
    #       f0      - pitch frequency
    '''
    if not winLength:
        winLength = xcorrWinLen(PARAMS, wav,fs)    
    zf = zeroFreqFilter(wav, fs, winLength)
    gci = np.squeeze(np.where(np.diff((zf>0).astype(int))==1))   #+ve zero crossings
    es = np.abs(zf[gci+1]-zf[gci-1])
    T0 = np.diff(gci)
    T0 = T0/fs
    f0 = 1/T0
    f0 = np.append(f0, f0[-1])
    return zf,gci,es,f0
	


def zff_computefn(PARAMS, Speech,fs):
    '''
    # Input:
    #        sample = Speech Signal
    #        fs = Sampling rate
    #
    # Output:
    '''
    Speech = np.divide(Speech, (1.01*np.max(np.abs(Speech)))+1e-10)
    winlength = 16
    zsp1, gclocssp1, epssp1, f0sp1 = zfsig2(PARAMS, Speech, fs, winlength)
    zsp1 = np.divide(zsp1, (np.max(np.abs(zsp1)))+1e-10)
    epssp1 = np.divide(epssp1, (np.max(epssp1))+1e-10)
    #epochstr = epssp1    # excitation strength at GCIs
    meanessp1 = np.mean(epssp1)    # mean value of excitation strength at GCIs
    epstrsp1 = [0]*len(Speech)
#    print('zff_computefn epstrsp1: ', np.shape(epstrsp1), np.shape(gclocssp1), np.shape(epssp1))
    for idx in range(len(gclocssp1)):
        epstrsp1[gclocssp1[idx]] = epssp1[idx]
    
    #voicing decision
    epssp1 = np.array((epssp1>0.7*meanessp1))
    
    vgclocssp1 = gclocssp1[epssp1]
    vf0sp1 = f0sp1[epssp1]
    
    return zsp1, epstrsp1, vgclocssp1, vf0sp1




def norm_auto(PARAMS, Xin, fs):
    Nframesize	= int(PARAMS['Tw'] * fs / 1000)
    Nframeshift	= int(PARAMS['Ts'] * fs / 1000)
    Nspeech = len(Xin)
    nFrames = int(np.floor((Nspeech-Nframesize)/Nframeshift)+1)
    pitch_period = np.zeros(nFrames)
    NormAutoCorr = np.zeros((nFrames,Nframesize))
    Pitch_Contour = np.zeros(nFrames)
    First_Peak = np.zeros(nFrames)
    numCoeffs = Nframesize
#    frameIdx = list(range(0, (Nspeech-Nframesize), Nframeshift))

    frames = librosa.util.frame(Xin, Nframesize, Nframeshift, axis=0)
#    for j in range(len(frameIdx)):
    for j in range(np.shape(frames)[0]):
#        i = frameIdx[j]
#        SpFrm = Xin[i:np.min([i+Nframesize, len(Xin)])]
        SpFrm = frames[j, :]

        #        print('norm_auto SpFrm: ', np.shape(SpFrm), SpFrm)
        a = np.correlate(SpFrm, SpFrm, 'full')
        a /= np.max(a)+1e-10
#        print('norm_auto a: ', np.shape(a), a)
                
        idx = np.array(list(range(int(np.ceil(len(a)/2)), int(np.floor(len(a)/2+numCoeffs)))))
#        print('norm_auto idx: ', np.shape(idx))
        if numCoeffs < len(idx):
            acf = a[idx[:numCoeffs]]
        elif numCoeffs == len(idx):
            acf = a[idx]
        else:
            acf = np.append(a[idx], np.zeros(np.abs(len(idx)-numCoeffs)))

        norm_auto_val = acf[16:]
#        print('norm_auto norm_auto_val: ', np.shape(norm_auto_val), norm_auto_val)
        FL, Properties = scipy.signal.find_peaks(norm_auto_val)
#        print('norm_auto FL: ', np.shape(FL))

        if np.size(FL)<1:
            Amp = norm_auto_val[0]
            val_peak = Amp
            period = (0+16)/fs
#            print('norm_auto FL=[] val_peak period: ', val_peak, period)
        elif np.size(FL)==1:
            Amp = norm_auto_val[FL.flatten()]
            val_peak = Amp
            period = (FL+16)/fs
#            print('norm_auto FL=[0] val_peak period: ', val_peak, period)
        else:
            Amp = norm_auto_val[FL]
            val_peak = np.max(Amp)
            in_val = np.argmax(Amp)
#            print('norm_auto in_val: ', in_val)
            FL[0] = FL[in_val]
            period = (FL[0]+16)/fs
#            print('norm_auto FL=[...] val_peak period: ', val_peak, period)
        
        First_Peak[j] = val_peak
        pitch_period[j] = period
        Pitch_Contour[j] = 1/period
        NormAutoCorr[j,:] = acf
    
    return NormAutoCorr, Pitch_Contour, First_Peak, pitch_period




def ResFilter_v2(PrevSpFrm, SpFrm, FrmLPC, LPorder, FrmSize):
    '''
    #USAGE: [ResFrm]=ResFilter_v2(PrevSpFrm,SpFrm,FrmLPC,LPorder,FrmSize,plotflag)
    '''
    ResFrm = np.zeros(FrmSize)
    tempfrm = np.zeros(FrmSize+LPorder)
    tempfrm[:LPorder] = PrevSpFrm
    tempfrm[LPorder:LPorder+FrmSize] = SpFrm[:FrmSize]
    
    for i in range(FrmSize):
        t = 0
        for j in range(LPorder):
            t += FrmLPC[j+1]*tempfrm[-j+i+LPorder]
        ResFrm[i] = SpFrm[i] - (-t)    
    return ResFrm



def LPres(speech, fs, framesize, frameshift, lporder, preemp):
    '''
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # USAGE : [res,LPCs,RSE] = 
    #		LPres(speech,fs,framesize,frameshift,lporder,preemp)
    #
    # INPUTS	:
    #	speech 		- speech signal (Nx1)
    #	fs		- sampling rate (in Hz)
    #	framesize 	- framesize for LP analysis (in ms)
    #	frameshift 	- frameshift for LP analysis (in ms)
    #	lporder		- order of LP analysis
    #	preemp		- If 0, no preemphasis is done, otherwise an high-pass 
    #			filtering is performed as per the following difference
    #			eqn:	y(n) = x(n) - a x(n-1),
    #			where 'x' is the speech signal and 'a = preemp'. 
    # 
    # OUTPUTS :
    #	res	- residual signal (Nx1)
    #	LPCs	- 2D array (M x p) containing LP coeffs of all frames,
    #		where p is the LP order and M depends on the framesize and
    #		frameshift used.
    #	RSE	- Residual to signal energy ratio
    #
    # This matlab function generates LP residual by inverse filtering. The inverse
    # filter coefficients are obtained by LPanalysis.
    #
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # METHOD: 
    #	- Consider one frame at a time, compute LP coefficients of the frame
    #	samples that are windowed using a hamming window.
    #	- Inverse filter the unwindowed speech frame to get corresponding 
    #	reisdual frame.
    #	- Only FRAMESHIFT number of samples among the total 
    #        FRAMESIZE number of samples are retained.
    #	-Repeat same procedure for all the frames. The last frame can be of
    #	size less than FRAMESIZE.
    #	-In the end, hamming window lporder number of 
    #        samples of the first frame 
    #	 to eliminate initial poorly predicted samples.
    #
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # ACKNOWLEDGEMENT	:
    #	This program has been adopted from LPresidual_v3.m authored by
    #	S.R.Mahadeva Prasanna.
    #
    # AUTHOR 		: Dhananjaya N
    # DATE   		: 22/11/2004
    # LAST MODIFIED 	: -
    # OTHER FUNCTIONS USED	: ResFilter_v2
    #
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # SOME MISTAKES IN THE PREVIOUS VERSIONS :
    #	res  = filter(lpcoef,1,temp1); %inverse filtering
    # NOTE the error in the above statement. we need to filter orginal frame, 
    # not the windowed one (temp1 is original frame).
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # LOG OF CHANGES:
    #
    #	Changes by S.R.M.Prasanna:
    # 
    # 24-10-2000: The program was not working for framshift=one sample. The index
    #	      values are properly adjusted and now it works for this case also.
    # 07-04-2002: The program was having bug in the routine of transferring
    #            'Previous Frame' values to ResFilter.m. Now it has been modified
    #
    #	Changes by Dhananjaya N:
    #
    # 22/11/2004: 	* The program was restructured. 
    #
    #		* The binary 'preempflag' input 
    #		parameter has been changed to a continuous parameter 'preemp
    #		and can take values from 0 to 1.
    #
    #		* A new i/p parameter 'fs' (sampling rate) is added.
    #
    #		* The units of the parameters framesize and frameshift are 
    #		changed to 'ms' from 'num of samples'.
    #
    #		* Zero energy frames are handled. Otherwise gives divide by 
    #		zero error.
    #
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    res = []
    LPCs = []
    RSE = []

    # Validating input arguments
    if framesize>50:
        print('!!! Ensure that the framesize and frameshift are in ms and not in terms of number of samples')
    
    #Converting unit of variuos lengths from 'time' to 'sample number'
    Nframesize = int(framesize * fs / 1000)
    Nframeshift = int(frameshift * fs / 1000)
    
    '''
    #---------------------------------------------------------------------
    # Added by Mrinmoy to remove extra samples at the end if less than
    # frame shift
    #     rem = mod((length(speech)-Nframesize),Nframeshift)
    #     if(rem < Nframeshift)
    #         speech = speech(1:end-rem);
    #     end
    #---------------------------------------------------------------------
    '''
    Nspeech = len(speech)
    
    # PREEMPHASIZING SPEECH SIGNAL
    if preemp!= 0:
        speech = librosa.effects.preemphasis(speech, coef=1)
    
    #COMPUTING RESIDUAL
    res = np.zeros(Nspeech)
    
    #NUMBER OF FRAMES
    nframes = int(np.floor((Nspeech-Nframesize)/Nframeshift)+1)
    
    LPCs = np.zeros((nframes,(lporder+1)))
    
    j = 0
    frames = librosa.util.frame(speech, Nframesize, Nframeshift, axis=0)
    for i in range(np.shape(frames)[0]):
        SpFrm = frames[i, :]
        # print('SpFrm energy: ', np.sum(np.abs(SpFrm)))
        if np.sum(np.abs(SpFrm))<1e-5:    #Handling zero energy frames
            LPCs[j,:] = np.zeros(lporder+1)
            ResFrm = np.zeros(Nframesize)
        else:
            try:
                lpcoef = librosa.core.lpc(np.multiply(scipy.signal.blackman(Nframesize), SpFrm), lporder)
                LPCs[j,:] = np.real(lpcoef)
            except:
                lpcoef = np.zeros(lporder+1)
                LPCs[j,:] = np.real(lpcoef)
                ResFrm = np.zeros(Nframesize)
            if (i*Nframeshift)<=lporder:
                PrevFrm = np.zeros(lporder)
            else:
                PrevFrm = speech[(i*Nframeshift-lporder):i*Nframeshift]   
            ResFrm	= ResFilter_v2(np.real(PrevFrm), np.real(SpFrm), np.real(lpcoef), lporder, Nframesize)
        res[i*Nframeshift:(i+1)*Nframeshift] = ResFrm[:Nframeshift]
        j += 1

    '''
    # The residual samples of the last but one frame is copied in entirity
    # This line is commented on 22/11/2004 by Dhananjaya.
    # A carefull analysis reveals that this will not be required. 
    # The handling of last frame takes care of this.
    '''
    #res[i+Nframeshift:i+Nframesize] = ResFrm[Nframeshift:Nframesize]
    
    #PROCESSING LASTFRAME SAMPLES
    #if i<Nspeech:
    #    SpFrm = speech[i:Nspeech]
    #    if np.sum(np.abs(SpFrm))==0:    #Handling zero energy frames
    #        LPCs[j,:] = 0
    #        ResFrm = np.zeros(Nframesize)
    #    else:
    #        lpcoef = librosa.core.lpc(np.multiply(scipy.signal.blackman(len(SpFrm)), SpFrm), lporder)
    #        LPCs[j,:] = np.real(lpcoef)
    #        PrevFrm	= speech[(i-lporder):i]
    #        ResFrm = ResFilter_v2(np.real(PrevFrm), np.real(SpFrm), np. real(lpcoef), lporder, Nframesize, 0)
    #        
    #    res[i:i+length(ResFrm)]	= ResFrm[len(ResFrm)]
    #    j += 1
        
    hm = np.hamming(2*lporder)
    for i in range(int(len(hm)/2)):
        res[i] = res[i] * hm[i]    #attenuating first lporder samples
    
    return res, LPCs, RSE



def remTrend2(h, sf):
    '''
    # USAGE : ret = RemTrend(henv,sf,plotflag);
    '''
    pd = int(np.floor(sf*2/1000))
    Ret = np.zeros(len(h))
    for i in range(pd):
        temp = h[:i+pd]
        Ret[i] = h[i]/(np.mean(temp)+1e-10)
        Ret[i] = Ret[i]*h[i]
        
    for i in range(pd, len(h)-pd):
        temp1 = h[i-pd:i+pd]
        Ret[i] = h[i]/(np.mean(temp1)+1e-10)
        Ret[i] = Ret[i]*h[i]
    
    for i in range(len(h)-pd, len(h)):
        temp2 = h[i:len(h)]
        Ret[i] = h[i]/(np.mean(temp2)+1e-10)
        Ret[i] = Ret[i]*h[i]
    
    Ret[np.squeeze(np.where(np.isnan(Ret)))] = np.min(Ret)

    return Ret


def HilbertEnv(sig, Fs, RemTrendFlag=0):
    '''
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # USAGE	: henv = HilbertEnv(sig,Fs,RemTrendFlag)
    #
    # PURPOSE	:
    #	Computes the magnitude of the analytic signal, obtained by taking the
    #	Hilbert transform of the i/p signal. Hilbert transform of a signal 
    #	shifts the phase of every freq. component of the signal by 90 degree.
    #	The matlab function hilbert() returns a complex analytic signal, whose
    #	real part is same as the input signal and its imaginary part is the 
    #	Hilbert transform of the signal.
    # 	Hilbert envelope is given by h_e[n] = sqrt(s[n]^2 + h[n]^2);
    #
    #	The magnitude of the analytic signal has the non-glottal closure 
    #	regions (valleys between the glottal peaks) raising significantly from 
    #	the zero line. The valleys are brought down, while at the same time, 
    #	the glottal closure (GC) peaks are sharpened using RemTrend() function. 
    #	This RemTrend() fn may be useful for detection of instants of GC and 
    #	pitch detection.
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    # Smooth the signal by 3 point hanning window for reducing noise
    #sig	= np.convolve(sig,hanning(3));
    #sig	= sig(2:length(sig)-1);
#    print('HilbertEnv sig: ', np.shape(sig), np.shape(scipy.signal.hilbert(sig)))
    hcmplx = scipy.signal.hilbert(sig)
    henv = np.abs(hcmplx)
    if RemTrendFlag==1:
#        print('HilbertEnv remTrend2: ', np.shape(hcmplx), np.shape(henv), np.shape(remTrend2(henv, Fs)))
        henv = remTrend2(henv, Fs)  #use 2ms window length    
    return henv



def peak_to_side_var(Xin, fs, Tw, Ts, EpochStrength):
    '''
    LPRES AND HENV
    '''
    LPorder = int(fs/1000)+4
    preemp = 0
    LPresidual, LPCs, RSE = LPres(Xin, fs, Tw, Ts, LPorder, preemp)
#    print('peak_to_side_var LPresidual: ', np.shape(LPresidual))
    hilbert_envelope = HilbertEnv(LPresidual, fs, 1)
    hilbert_envelope = hilbert_envelope/(1.05*np.max(hilbert_envelope)+1e-10)
    #Nframe_size = int(Tw*fs/1000)
    #Nframe_shift = int(Ts*fs/1000)
    #Nspeech = len(Xin)
    #nFrames = int(np.floor((Nspeech-Nframe_size)/Nframe_shift)+1)
    j=0
    hilbert_envelope = hilbert_envelope/(1.05*np.max(hilbert_envelope)+1e-10)
    EpochStrength = EpochStrength/(1.05*np.max(EpochStrength)+1e-10)
    ff = np.squeeze(np.where(EpochStrength>0.01))
    frame_epoch_length = 12    # window for searching peaks around epoch locations (25 samples)
    frame_side_lobe = 30    # window for computing side lobe variance (20 samples)
    PSR = np.zeros(len(ff))
    peak_val_arr = np.zeros(len(ff))
    index_peak_arr = np.zeros(len(ff))
    for i in range(len(ff)):
        epoch_loc = ff[i]
        frame_epoch_length_left = epoch_loc-frame_epoch_length
        frame_epoch_length_right = epoch_loc+frame_epoch_length
        henv_l2rfrm = hilbert_envelope[frame_epoch_length_left:frame_epoch_length_right]
        FL, Properties = scipy.signal.find_peaks(henv_l2rfrm)

        if np.size(FL)<1:
            Amp = henv_l2rfrm[0]
            peak_val = Amp
            peak_loc = 0
        elif np.size(FL)==1:
            FL = int(FL)
            Amp = henv_l2rfrm[FL]
            peak_val = Amp
            peak_loc = FL
        else:
            Amp = henv_l2rfrm[FL]
            peak_val = np.max(Amp)
            ind_loc = np.argmax(Amp)
            peak_loc = FL[ind_loc]
        
#        print('peak_to_side_var FL: ', FL)
        
        if peak_val>=0.00:
            if peak_loc>=(frame_epoch_length+1):
                act_peak_loc = (peak_loc-(frame_epoch_length+1))+epoch_loc
            else:
                act_peak_loc = epoch_loc-((frame_epoch_length+1)-peak_loc)
            index_peak = act_peak_loc
            peak_val_arr[i] = peak_val
            index_peak_arr[i] = index_peak.astype(int)
            s_left = (index_peak-1)-frame_side_lobe
            s_right = (index_peak+1)+frame_side_lobe
            if s_left<=0:
                PSR[i] = 0
                continue
            elif s_right>len(hilbert_envelope):
                PSR[i] = 0
                continue
            else:
#                print('peak_to_side_var s_left: ', np.shape(hilbert_envelope), s_left, index_peak)
                frame_left = hilbert_envelope[s_left:index_peak]
                frame_right = hilbert_envelope[index_peak:s_right]
                side_lobe = [frame_left, frame_right]
                side_lobe_var = np.var(side_lobe) #var(side_lobe,1)
                if side_lobe_var==0:
                    PSR[i] = peak_val
                    continue
                PSR[i] = np.divide(peak_val, side_lobe_var+1e-10)
        else:
            PSR[i] = 0
    Peak_side_lobe = np.divide(PSR, np.max(PSR)+1e-10)
    mm = np.zeros(len(hilbert_envelope))
    index_peak_arr = np.array(index_peak_arr).astype(int)
#    print('peak_to_side_var index_peak_arr: ', index_peak_arr[0], np.shape(index_peak_arr))
#    print('peak_to_side_var index_peak_arr: ', index_peak_arr)
    mm[:index_peak_arr[0]] = Peak_side_lobe[0]    #inter_psr
    j = 1
    for i in range(index_peak_arr[0]+1, len(hilbert_envelope)):
        if i>index_peak_arr[-1]:
            mm[i] = Peak_side_lobe[-1]
            continue
        if i>=index_peak_arr[j]:
            mm[i] = Peak_side_lobe[j]
            j += 1
            continue
        else:
            mm[i] = Peak_side_lobe[j]

    Peak_side_lobe_inter = mm
    return index_peak_arr, hilbert_envelope, peak_val_arr, Peak_side_lobe_inter, PSR




def compute_Khonglah_et_al_features(PARAMS, Xin, fs):
    Nframesize = int(PARAMS['Tw']*fs/1000)
    Nframeshift = int(PARAMS['Ts']*fs/1000)
    FV = np.empty([])

    ZFFS, EpochStrength, Voiced_GCI, VoicedPitch = zff_computefn(PARAMS, Xin, fs)
    # print('ZFFS: ', np.shape(ZFFS))
    
    NormAutoCorr, Pitch_Contour, First_Peak, pitch_period = norm_auto(PARAMS, ZFFS, fs)
    # print('\tFirst_Peak: ', np.shape(First_Peak))
    FV = np.array(First_Peak, ndmin=2).T
    
    ms1, ModSpec = modspec.modulationspectrum(Xin, fs, PARAMS['NBANDS'], PARAMS['NORDER'], PARAMS['LPFFc'], Fmin=0, Fmax=fs/2, WL=PARAMS['Tw'], OLN=PARAMS['Ts'])
    if len(ModSpec)<len(Xin):
        ModSpec = np.append(ModSpec, np.zeros(len(Xin)-len(ModSpec)))
    else:
        ModSpec = ModSpec[:len(Xin)]
    ModSpec_frames = librosa.util.frame(ModSpec, Nframesize, Nframeshift, axis=0)
    ModSpec_feat = np.mean(ModSpec_frames, axis=1)
    # print('\tModSpec_feat: ', np.shape(Xin), np.shape(ModSpec), np.shape(ModSpec_feat), np.shape(FV))
    FV = np.append(FV, np.array(ModSpec_feat, ndmin=2).T, axis=1)
    
    index_peak_arr, hilbert_envelope, peak_val_arr, PSR_inter, PSR = peak_to_side_var(Xin, fs, PARAMS['Tw'], PARAMS['Ts'], EpochStrength)
    PSR_frames = librosa.util.frame(PSR_inter, Nframesize, Nframeshift, axis=0)
    PSR_feat = np.mean(PSR_frames, axis=1)
    # print('\tPSR_feat: ', np.shape(PSR_feat))
    FV = np.append(FV, np.array(PSR_feat, ndmin=2).T, axis=1)
    
    ZCR = librosa.feature.zero_crossing_rate(y=Xin, frame_length=Nframesize, hop_length=Nframeshift, center=False)
    # print('\tZCR: ', np.shape(ZCR))
    FV = np.append(FV, np.array(ZCR, ndmin=2).T, axis=1)

    Rms = librosa.feature.rms(y=Xin, frame_length=Nframesize, hop_length=Nframeshift, center=False)
    Energy = np.power(Rms,2)
    # print('\tEnergy: ', np.shape(Energy))
    FV = np.append(FV, np.array(Energy, ndmin=2).T, axis=1)
        
    Mel_FBE = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, n_mels=PARAMS['no_filt'], center=False)
    LogMelEnergy = np.sum(np.log10(Mel_FBE[:18, :]), axis=0)
    # print('\tLogMelEnergy: ', np.shape(LogMelEnergy), np.shape(Mel_FBE))
    FV = np.append(FV, np.array(LogMelEnergy, ndmin=2).T, axis=1)

    Spectral_Centroid = librosa.feature.spectral_centroid(y=Xin, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, center=False)
    # print('\tSpectral_Centroid: ', np.shape(Spectral_Centroid))
    FV = np.append(FV, np.array(Spectral_Centroid, ndmin=2).T, axis=1)
    
    Spectral_Flux = spflux.SpectralFlux(Xin, fs, PARAMS['Tw'], PARAMS['Ts'])
    # print('\tSpectral_Flux: ', np.shape(Spectral_Flux))
    FV = np.append(FV, np.array(Spectral_Flux, ndmin=2).T, axis=1)
    
    Spectral_Rolloff = librosa.feature.spectral_rolloff(y=Xin, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, roll_percent=0.8, center=False)
    # print('\tSpectral_Rolloff: ', np.shape(Spectral_Rolloff))
    FV = np.append(FV, np.array(Spectral_Rolloff, ndmin=2).T, axis=1)

    PercentLowEnergyFrms = np.sum(Energy<0.5*np.mean(Energy))/np.shape(Energy)[1]
    PercentLowEnergyFrms_feat = np.repeat(np.array(PercentLowEnergyFrms, ndmin=2), np.shape(Energy)[1], axis=1)
    # print('\tPercentLowEnergyFrms_feat: ', np.shape(PercentLowEnergyFrms_feat))
    FV = np.append(FV, np.array(PercentLowEnergyFrms_feat, ndmin=2).T, axis=1)
    
    return FV.astype(np.float32)
    
