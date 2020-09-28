#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:41:36 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import scipy
import scipy.fftpack
import scipy.ndimage




def modified_group_delay_feature(frames, rho=0.4, gamma=0.9, num_coeff=12, frame_length=0.025, frame_shift=0.010):
    '''
    # input: 
    #     file_name: path for the waveform. The waveform should have a header
    #     rho: a parameter to control the shape of modified group delay spectra
    #     gamma: a parameter to control the shape of the modified group delay spectra
    #     num_coeff: the desired feature dimension
    #     [frame_length]: 
    #     [frame_shift]: 
    #
    # output:
    #     grp_phase: modifed gropu delay spectrogram
    #     cep: modified group delay cepstral feature.
    #     ts: time instants at the center of each analysis frame.
    #
    # Example:
    #     [grp_phase, cep, ts] = modified_group_delay_feature('./100001.wav', 0.4, 0.9, 12);
    # Please tune rho and gamma for better performance
    #     See also: howtos/HOWTO_features.m   
    #
    # by Zhizheng Wu (zhizheng.wu@ed.ac.uk)
    # http://www.zhizheng.org
    #
    # The code has been used in the following three papers:
    # Zhizheng Wu, Xiong Xiao, Eng Siong Chng, Haizhou Li, "Synthetic speech detection using temporal modulation feature", IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2013.
    # Zhizheng Wu, Tomi Kinnunen, Eng Siong Chng, Haizhou Li, Eliathamby Ambikairajah, "A study on spoofing attack in state-of-the-art speaker verification: the telephone speech case", Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) 2012. 
    # Zhizheng Wu, Eng Siong Chng, Haizhou Li, "Detecting Converted Speech and Natural Speech for anti-Spoofing Attack in Speaker Recognition", Interspeech 2012. 
    #
    # feel free to modify the code and welcome to cite above papers :)
    #
    # [speech,fs]  = audioread(file_name);
    # 
    # % Removing the silences -- Added by Mrinmoy
    # [speech, silPos] = remove_silence(speech, frame_length*1000, frame_shift*1000, fs, 0.05);
    '''

    NFFT = 512;
    
    frame_num = np.shape(frames)[0]
    frame_length = np.shape(frames)[1]
    delay_vector = np.array(list(range(frame_length)), ndmin=2)
    delay_matrix = np.repeat(delay_vector, frame_num, axis=0)
    
    delay_frames = np.multiply(frames, delay_matrix)
    
    x_spec = scipy.fftpack.fft(frames, NFFT).T
    y_spec = scipy.fftpack.fft(delay_frames, NFFT).T
#    print('mgd x-y-spec: ', np.shape(x_spec), np.shape(y_spec))

    x_spec = x_spec[:int(NFFT/2+1), :]
    y_spec = y_spec[:int(NFFT/2+1), :]
    
    temp_x_spec = np.abs(x_spec)
    
    dct_spec = scipy.fftpack.dct(scipy.ndimage.median_filter(np.log(temp_x_spec), 5))
    smooth_spec = scipy.fftpack.idct(dct_spec[:30,:], n=int(NFFT/2+1), axis=0).astype(np.float128)
    
    # grp_phase1 = np.divide((np.multiply(np.real(x_spec),np.real(y_spec)) + np.multiply(np.imag(y_spec),np.imag(x_spec))), np.power(np.exp(smooth_spec), (2*rho)) + 1e-10 )
    grp_phase1_deno = np.power(np.exp(smooth_spec).astype(np.float32), (2*rho)).astype(np.float32)
    grp_phase1 = np.divide((np.multiply(np.real(x_spec),np.real(y_spec)) + np.multiply(np.imag(y_spec),np.imag(x_spec))), grp_phase1_deno + 1e-10 )
    
    # grp_phase = np.multiply(np.divide(grp_phase1, np.abs(grp_phase1)+1e-10) , np.power(np.abs(grp_phase1), gamma))
    grp_phase = np.multiply(np.divide(grp_phase1, np.abs(grp_phase1)+1e-10) , np.power(np.abs(grp_phase1), gamma).astype(np.float32))

    grp_phase = np.divide(grp_phase, (np.max(np.max(np.abs(grp_phase))))+1e-10)
    
    grp_phase[np.isnan(grp_phase)] = 0.0
    grp_phase = grp_phase[-1::-1, :]
    
    cep = scipy.fftpack.dct(grp_phase, n=num_coeff, axis=0).T
#    print('mgd cep: ', np.shape(cep))
    
    return grp_phase, cep



def compute_mgdcc(PARAMS, frames, fs):
    rho = 0.4
    gamma = 0.9
    num_coeff = PARAMS['n_cep']
    frame_length = PARAMS['Tw']/1000 # in secs
    frame_shift = PARAMS['Ts']/1000 # in secs
    grp_phase, MGDCC = modified_group_delay_feature(frames, rho, gamma, num_coeff, frame_length, frame_shift)
    
    return grp_phase, MGDCC
