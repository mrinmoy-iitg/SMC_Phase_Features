#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 10:04:36 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import librosa



def next_pow_2(n):
    npow = 1
    while npow<n:
        npow = npow*2
    return npow



def ngdpreprocbyzeroatpi(x,fs,N):
    #n1ms = int(fs/1000)    
    if len(x)>N:
        x= x[:N]
    else:
        x = np.append(x, np.zeros(N-len(x)))
    n = np.pi*np.array(list(range(N)))/(N+1e-10)
    w = (2*(1+scipy.cos(n)))
    xw = np.multiply(w, x)
    return xw,w



def ngdpreprocbypoleat0(x,fs,N):
    #weight by 1/(4*sin^2(w)) , assume N=20 ms; eg: 320 at 16kHz.
    #n1ms = int(fs/1000)
    if len(x)>N:
        x= x[:N]
    else:
        x = np.append(x, np.zeros(N-len(x)))
    n = np.pi*np.array(list(range(N)))/(N+1e-10)
    w = 1/(2*(1-scipy.cos(n))+1e-10)
    w[0] = 0
    
    xw = np.multiply(w.flatten(), x.flatten())
    
    return xw,w




def gdelay1d(x, Nfft, fs=8000, m=0.5, **kwargs):
    '''
    Usage: [gd,f,ngd,mgd,dmag] = gdelay1d(x,Nfft,fs,m)
    
    Computes the group delay (GD) spectrum of a 1D signal. 
    
    Inputs:
    x 	- input signal
    Nfft	- determines the FFt length or the number of samples in the frequency 
        domain over the range [0 2*pi] (default Nfft=length(x))
    fs	- sampling rate if 'f' the frequency indices at which the GD spectrum
        is sampled.
    m	- modification factor for computing the modified group delay
        (default m=0.5)
    
    Outputs:
    gd	- conventional group delay,
        tau(w) = -d{theta(w)}/dw = Re{Y./X} = (Xr.*Yr + Xi.*Yi)./(|X|.^2)
        where Y is the DFT of the signal y[n] = n * x[n];
    f	- frequency indices at which the GD spectrum is sampled.
    ngd	- numerator of the group delay,
        tau_n(w) = tau(w) * |X|.^2 = (Xr.*Yr + Xi.*Yi)
    mgd	- modified group delay,
        tau_m(w) = (Xr.*Yr + Xi.*Yi)./(|X|.^m)
    where 'm' is the modification factor to condition the denominator
    '''
    f = np.array(list(range(Nfft)))*fs/Nfft
    # Check for zero signal
    if np.sum(np.power(x, 2))==0:
        gd = np.zeros(Nfft)
        ngd = np.zeros(Nfft)
        mgd = np.zeros(Nfft)
        return gd, f, ngd, mgd
    
#    x = x.flatten()
    X = scipy.fftpack.fft(x,Nfft)
    
    Nx = np.size(x)
    n = np.array(list(range(Nx)))
#    n = n.flatten()
    
#    print(np.shape(n), np.shape(x))
    
    y = np.multiply(n, x)
    Y = scipy.fftpack.fft(y, Nfft)
    
    Xr = np.real(X)
    Xi = np.imag(X)
    Xm = np.abs(X)
    if np.sum(Xm==0)>0:
        print('Magnitude spectrum is zero at some frequencies. Fixing divide by zero error')
        Xm[np.squeeze(np.where(Xm==0))] = np.min(Xm[np.squeeze(np.where(Xm!=0))])
        
    Yr = np.real(Y)
    Yi = np.imag(Y)
    
    ngd = np.multiply(Xr,Yr) + np.multiply(Xi, Yi)
    
    gd = np.power(np.divide(ngd, Xm+1e-10), 2)

    mgd = np.power(np.divide(np.abs(ngd), np.abs(Xm+1e-10)), m)
    mag = Xm
    
    return gd, f, ngd, mgd, mag



def magspectrum(x,fs,nfft):
    '''
    Usage: [Xm,f] = magspectrum(x,fs,nfft)
    '''
    X = scipy.fftpack.fft(x,nfft)
    Xm = np.abs(X)
    Xm = Xm[:nfft/2+1]
    f = list(range(nfft/2))*fs/nfft
    return Xm,f



def ztl(x,fs,nfft,nwin):
    '''
    Usage: [hngd,f,dngd,ngd,mag,hgd,dgd,gd] = ztl(x,fs,nfft,nwin,PLOTFLAG)
    
    Preemphasis assumed to be done a priori
    [s,fs]=wavread('~/myrecordings/timit-train-dr1-mgrl0-sa2.wav');
    s=resample(s,10000,fs);
    fs=10000;
    n=2500;for i=n+[1:1000];x=s(i:i+50-1);[hngd,f,dngd,ngd,mag,hgd,dgd,gd] = ztl(x,fs,nfft,nwin,1);pause;end;
    '''    
    #n1ms = int(fs/1000)
    nfftby2 = int(np.floor(nfft/2))
    #Nx = len(x)
    
#    print('x: ', np.shape(x))
    xw, w1 = ngdpreprocbyzeroatpi(x, fs, nwin)
#    print('xw: ', np.shape(xw), fs, nwin)
    xw, w2 = ngdpreprocbypoleat0(xw, fs, nwin)
#    print('xw: ', np.shape(xw), fs, nwin)
    xw, w3 = ngdpreprocbypoleat0(xw, fs, nwin)
#    print('xw: ', np.shape(xw), fs, nwin)
    
    #w = np.power(np.multiply(w1[:nwin], w2[:nwin]), 2)
    
    m = 0.5
    gd__, f, ngd__, mgd, mag = gdelay1d(xw, nfft, fs, m)
    
    ngd__ = ngd__.tolist()
    ngd = ngd__
    ngd.extend(ngd__)
    ngd.extend(ngd__)
    
    gd__ = gd__.tolist()
    gd = gd__
    gd.extend(gd__)
    gd.extend(gd__)
    
    dngd = -np.diff(np.diff(ngd))
    dgd = -np.diff(np.diff(gd))
    
    hngd = np.abs(scipy.signal.hilbert(np.diff(dngd)))
    hgd = np.abs(scipy.signal.hilbert(np.diff(dgd)))
    
    f = f[:nfftby2+1]
    
    ngd = ngd[:nfftby2+1]
    gd = gd[:nfftby2+1]
    mag = mag[:nfftby2+1]
    
    dngd = dngd[nfft-2+np.array(list(range(nfftby2+1)))]
    dgd = dgd[nfft-2+np.array(list(range(nfftby2+1)))]
    
    hngd = hngd[nfft-4+np.array(list(range(nfftby2+1)))]
    hgd = hgd[nfft-4+np.array(list(range(nfftby2+1)))]
    
    return hngd,f,dngd,ngd,mag,hgd,dgd,gd


def compute_hngd(PARAMS, frames, frame_size, fs):
    nfft = next_pow_2(frame_size)*2
    n1ms = int(fs/1000)
    nwin = PARAMS['Tw']*n1ms
    
    HNGD = []
#    print('Xin: ', np.shape(Xin), frame_size, frame_shift)
    frames = frames.T
#    print('Framing done: ', np.shape(frames))
    
    HNGD = np.zeros((np.shape(frames)[1], int(np.ceil(nfft/2))))
#    print('HNGD: ', np.shape(HNGD), np.shape(frames))
    for i in range(np.shape(frames)[1]):
        x_frame = frames[:,i]
#        print('x_frame: ', np.shape(x_frame))
        hngd, f, dngd, ngd, mag, hgd, dgd, gd = ztl(x_frame, fs, nfft, nwin)
#        print('HNGD: ', np.shape(HNGD), np.shape(hngd[:-1]))
        HNGD[i,:] = hngd[:-1]
    HNGD = HNGD[:,-1::-1].T    #All frames as columns    

    HNGD_MFCC, HNGD_FBE = hngd_mfcc(PARAMS, HNGD, fs)
    
    return HNGD, HNGD_FBE, HNGD_MFCC
    


def hz2mel(hz):
    '''
    Forward and backward mel frequency warping (see Eq. (5.13) on p.76 of [2]) 
    Note that base 10 is used in [2], while base e is used here and in HTK code
    '''
    return 1127*np.log(1+hz/700) #Hertz to mel warping function


def mel2hz(mel):
    '''
    Forward and backward mel frequency warping (see Eq. (5.13) on p.76 of [2]) 
    Note that base 10 is used in [2], while base e is used here and in HTK code
    '''
    return 700*np.exp(mel/1127)-700 #mel to Hertz warping function


def trifbank(M, K, R, fs, **kwargs):
    '''
	TRIFBANK Triangular filterbank.

	[H,F,C]=TRIFBANK(M,K,R,FS,H2W,W2H) returns matrix of M triangular filters 
	(one per row), each K coefficients long along with a K coefficient long 
	frequency vector F and M+2 coefficient long cutoff frequency vector C. 
	The triangular filters are between limits given in R (Hz) and are 
	uniformly spaced on a warped scale defined by forward (H2W) and backward 
	(W2H) warping functions.

	Inputs
		M is the number of filters, i.e., number of rows of H

		K is the length of frequency response of each filter 
		  i.e., number of columns of H

		R is a two element vector that specifies frequency limits (Hz), 
		  i.e., R = [ low_frequency high_frequency ];

		FS is the sampling frequency (Hz)

		H2W is a Hertz scale to warped scale function handle

		W2H is a wared scale to Hertz scale function handle

	Outputs
		H is a M by K triangular filterbank matrix (one filter per row)

		F is a frequency vector (Hz) of 1xK dimension

		C is a vector of filter cutoff frequencies (Hz), 
		  note that C(2:end) also represents filter center frequencies,
		  and the dimension of C is 1x(M+2)

	Example
		fs = 16000;               % sampling frequency (Hz)
		nfft = 2^12;              % fft size (number of frequency bins)
		K = nfft/2+1;             % length of each filter
		M = 23;                   % number of filters

		hz2mel = @(hz)(1127*log(1+hz/700)); % Hertz to mel warping function
		mel2hz = @(mel)(700*exp(mel/1127)-700); % mel to Hertz warping function

		% Design mel filterbank of M filters each K coefficients long,
		% filters are uniformly spaced on the mel scale between 0 and Fs/2 Hz
		[ H1, freq ] = trifbank( M, K, [0 fs/2], fs, hz2mel, mel2hz );

		% Design mel filterbank of M filters each K coefficients long,
		% filters are uniformly spaced on the mel scale between 300 and 3750 Hz
		[ H2, freq ] = trifbank( M, K, [300 3750], fs, hz2mel, mel2hz );

		% Design mel filterbank of 18 filters each K coefficients long, 
		% filters are uniformly spaced on the Hertz scale between 4 and 6 kHz
		[ H3, freq ] = trifbank( 18, K, [4 6]*1E3, fs, @(h)(h), @(h)(h) );

		 hfig = figure('Position', [25 100 800 600], 'PaperPositionMode', ...
		                   'auto', 'Visible', 'on', 'color', 'w'); hold on; 
		subplot( 3,1,1 ); 
		plot( freq, H1 );
		xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 

		subplot( 3,1,2 );
		plot( freq, H2 );
		xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 

		subplot( 3,1,3 ); 
		plot( freq, H3 );
		xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 

	Reference
		[1] Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
		    A guide to theory, algorithm, and system development. 
		    Prentice Hall, Upper Saddle River, NJ, USA (pp. 314-315).

		[2] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., 
		    Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D., 
		    Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK 
		    Version 3.4.1). Engineering Department, Cambridge University.
		    (see also: http://htk.eng.cam.ac.uk)

	Original Author
		Kamil Wojcicki, June 2011
	Translated to python: Mrinmoy Bhattacharjee, July 2020
    '''
    f_min = 0       #filter coefficients start at this frequency (Hz)
    f_low = R[0]    #lower cutoff frequency (Hz) for the filterbank 
    f_high = R[1]   #upper cutoff frequency (Hz) for the filterbank 
    f_max = 0.5*fs    #filter coefficients end at this frequency (Hz)
    f = np.linspace(f_min, f_max, K)    #frequency range (Hz), size 1xK

    # filter cutoff frequencies (Hz) for all filters, size 1x(M+2)
    c = mel2hz(hz2mel(f_low)+np.array(list(range(0,M+2)))*((hz2mel(f_high)-hz2mel(f_low))/(M+2)))

    # implements Eq. (6.140) given in [1] (for a given set of warping functions)
    H = np.zeros((M, K))            #zero otherwise
    for m in range(M):
        k = np.logical_and([f[i]>=c[m] for i in range(K)], [f[i]<=c[m+1] for i in range(K)]) #up-slope
        count = 0
        for k1 in k:
            if k1:
#                print('Selected k: %d %.2f %.2f %.2f ' % (count, c[m], f[count], c[m+1]), k1)
                val = 2*(f[count]-c[m]) / ((c[m+2]-c[m])*(c[m+1]-c[m])+1e-10)
                H[m, count] = val
#                if val<0:
#                    print('Negative1: ', m, count, val)
            count += 1
        
        k = np.logical_and([f[i]>=c[m+1] for i in range(K)], [f[i]<=c[m+2] for i in range(K)]) #down-slope
        count = 0
        for k1 in k:
            if k1:
#                print('Selected k: %d %.2f  %.2f  %.2f ' % (count, c[m+1], f[count], c[m+2]), k1)
                val = 2*(c[m+2]-f[count]) / ((c[m+2]-c[m])*(c[m+2]-c[m+1])+1e-10)
                H[m, count] = val
#                if val<0:
#                    print('Negative2: ', m, count, val)
            count += 1
    
#    maxH = np.repeat(np.array(np.max(H, axis=1), ndmin=2).T, K, axis=1)+1e-10
#    H = np.divide(H, maxH)    #normalize to unit height
#    trapzH = np.repeat(np.array(np.trapz(f,H,1), ndmin=2).T, K, axis=1)
#    H = np.divide(H, trapzH)    #normalize to unit area (inherently done)
    
    return H, f, c



def dctm(N, M):
    '''
    Type III DCT matrix routine (see Eq. (5.14) on p.77 of [1])
    '''
    arr1 = np.repeat(np.array(list(range(N)), ndmin=2).T, M, axis=1)
    arr2 = np.repeat(np.divide(np.pi*np.array(list(range(1,M+1)), ndmin=2)-0.5, M+1e-10), N, axis=0)
    dct = np.sqrt(2.0/(M+1e-10)) * scipy.cos(np.multiply(arr1, arr2))
    return dct
    
    
def ceplifter(N, L):
    '''
    Cepstral lifter routine (see Eq. (5.12) on p.75 of [1])
    '''
    return 1 + 0.5*L*scipy.sin(np.multiply(np.pi,np.array(list(range(0,N)))/(L+1e-10)))


def hngd_mfcc(PARAMS, MAG, fs):
    '''
    Author: Mrinmoy Bhattacharjee, 24 January 2020
    Original function: MFCC computation (Dan Ellis code) from input speech signal. This
    function takes a magnitude spectrum and computes the mel filtered
    energies and their corresponding cepstrum coefficients

	References

		[1] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., 
		    Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D., 
		    Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK 
		    Version 3.4.1). Engineering Department, Cambridge University.
		    (see also: http://htk.eng.cam.ac.uk)

		[2] Ellis, D., 2005. Reproducing the feature outputs of 
		    common programs using Matlab and melfcc.m. url: 
		    http://labrosa.ee.columbia.edu/matlab/rastamat/mfccs.html

		[3] Huang, X., Acero, A., Hon, H., 2001. Spoken Language 
		    Processing: A guide to theory, algorithm, and system 
		    development. Prentice Hall, Upper Saddle River, NJ, 
		    USA (pp. 314-315).
    '''
    
    ## PRELIMINARIES 
    nfft = np.shape(MAG)[0]      # length of FFT analysis 
    K = int(nfft/2+1)            #length of the unique part of the FFT 
    R = [0, fs/2]


    ## FEATURE EXTRACTION 

    '''
    Triangular filterbank with uniformly spaced filters on mel scale
    '''
    H, f, c = trifbank(PARAMS['no_filt'], K, R, fs) # size of H is M x K 
    
    '''
    Filterbank application to unique part of the magnitude spectrum
    '''
    HNGD_FBE = np.matmul(H, MAG[:K,:]) # FBE( FBE<1.0 ) = 1.0; # apply mel floor

    '''
    DCT matrix computation
    '''
    DCT = dctm(PARAMS['n_cep'], PARAMS['no_filt'])

    '''
    Conversion of logFBEs to cepstral coefficients through DCT
    '''
    HNGDMFCC =  np.matmul(DCT, np.log(HNGD_FBE+1e-5))

    '''
    Cepstral lifter computation
    '''
    lifter = ceplifter(PARAMS['n_cep'], PARAMS['no_filt'])

    '''
    Cepstral liftering gives liftered cepstral coefficients
    '''
    HNGDMFCC = np.matmul(np.diag(lifter), HNGDMFCC).T # ~ HTK's MFCCs

    return HNGDMFCC, HNGD_FBE


