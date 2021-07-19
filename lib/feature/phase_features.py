#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:21:10 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import lib.feature.hngdmfcc as hngd
import lib.feature.mgdcc as mgd
import lib.feature.ifcc as ifcc
import time
import numpy as np
import librosa




def compute_phase_based_features(PARAMS, Xin, fs):
    '''
    The input audio is split into 1sec segments to compute these features
    because the IFCC computation process takes an extremely long time for long audio
    sequences
    '''
    HNGD = np.empty([])
    HNGD_FBE = np.empty([])
    HNGDMFCC = np.empty([])
    HNGDMFCC_D_DD = np.empty([])
    
    grp_phase = np.empty([])
    MGDCC = np.empty([])
    MGDCC_D_DD = np.empty([])
    
    IF_SPEC = np.empty([])
    IFCC = np.empty([])
    IFCC_D_DD = np.empty([])

    frame_size = int(PARAMS['Tw']*fs/1000)
    frame_shift = int(PARAMS['Ts']*fs/1000)
    frames = librosa.util.frame(Xin, frame_size, frame_shift, axis=0)
    nFrames = np.shape(frames)[0]
    
    timeTaken_hngdmfcc = 0
    timeTaken_mgdcc = 0
    timeTaken_ifcc = 0
    segment = 0
    segment_dur = 0

    frmStart = 0
    frmEnd = 0
    for frmStart in range(0, nFrames, PARAMS['intervalSize']):
        frmEnd = np.min([frmStart + PARAMS['intervalSize'], nFrames])
        frames_part = frames[frmStart:frmEnd, :]
        # print('Frames: ', np.shape(frames_part))
        
        startTime_hngdmfcc = time.time()
        HNGD_part, HNGD_FBE_part, HNGDMFCC_part = hngd.compute_hngd(PARAMS, frames_part, frame_size, fs)
        if np.size(HNGD)<=1:
            HNGD = HNGD_part
            HNGD_FBE = HNGD_FBE_part
            HNGDMFCC = HNGDMFCC_part 
        else:
            # print('HNGD: ', np.shape(HNGD), np.shape(HNGD_part))
            HNGD = np.append(HNGD, HNGD_part, axis=1)
            # print('HNGD_FBE: ', np.shape(HNGD_FBE), np.shape(HNGD_FBE_part))
            HNGD_FBE = np.append(HNGD_FBE, HNGD_FBE_part, axis=1)
            # print('HNGDMFCC_D_DD: ', np.shape(HNGDMFCC_D_DD), np.shape(HNGDMFCC_D_DD_part))
            HNGDMFCC = np.append(HNGDMFCC, HNGDMFCC_part , axis=0)
        endTime_hngdmfcc = time.time()
        timeTaken_hngdmfcc += endTime_hngdmfcc-startTime_hngdmfcc


        startTime_mgdcc = time.time()
        grp_phase_part, MGDCC_part = mgd.compute_mgdcc(PARAMS, frames_part, fs)
        if np.size(grp_phase)<=1:
            grp_phase = grp_phase_part
            MGDCC = MGDCC_part
        else:
            # print('grp_phase: ', np.shape(grp_phase), np.shape(grp_phase_part))
            grp_phase = np.append(grp_phase, grp_phase_part, axis=1)
            # print('MGDCC_D_DD: ', np.shape(MGDCC_D_DD), np.shape(MGDCC_D_DD_part))
            MGDCC = np.append(MGDCC, MGDCC_part, axis=0)
        endTime_mgdcc = time.time()
        timeTaken_mgdcc += endTime_mgdcc-startTime_mgdcc


        startTime_ifcc = time.time()
        Xin_part = []
        for i in range(np.shape(frames_part)[0]-1):
            Xin_part.extend(frames_part[i,:frame_shift])
        Xin_part.extend(frames_part[-1,:])
        Xin_part = np.array(Xin_part)
        # print('IFCC Xin_part: ', np.shape(Xin_part), np.shape(Xin))
        IF_SPEC_part, IFCC_part = ifcc.compute_ifcc(PARAMS, Xin_part, fs)        
        if np.size(IF_SPEC)<=1:
            IF_SPEC = IF_SPEC_part
            IFCC = IFCC_part
        else:
            # print('IF_SPEC: ', np.shape(IF_SPEC), np.shape(IF_SPEC_part))
            IF_SPEC = np.append(IF_SPEC, IF_SPEC_part, axis=0)
            # print('IFCC_D_DD: ', np.shape(IFCC_D_DD), np.shape(IFCC_D_DD_part))
            IFCC = np.append(IFCC, IFCC_part, axis=0)
        endTime_ifcc = time.time()                
        timeTaken_ifcc += endTime_ifcc-startTime_ifcc

        segment += 1
        segment_dur += (np.shape(frames_part)[0]/nFrames)*(len(Xin)/fs)
        print('\tSegment %.3d (%.2f sec/%.2f sec)\tTime taken: hngdmfcc=%.2f sec, mgdcc=%.2f sec, ifcc=%.2f sec' % (segment+1, segment_dur, len(Xin)/fs, timeTaken_hngdmfcc, timeTaken_mgdcc, timeTaken_ifcc), end='\r', flush=True)
    print(' ')

    
    delta_win_size = np.min([PARAMS['delta_win_size'], np.shape(HNGDMFCC)[0]])
    # print('compute_HNGDMFCC delta_win_size: ', delta_win_size, np.shape(HNGDMFCC))
    if delta_win_size%2==0: # delta_win_size must be an odd integer >=3
        delta_win_size = np.max([delta_win_size-1, 3])
    if np.shape(HNGDMFCC)[0]<3:
        HNGDMFCC = np.append(HNGDMFCC, HNGDMFCC, axis=0)
    D_HNGD_MFCC = librosa.feature.delta(HNGDMFCC, width=delta_win_size, axis=0)
    DD_HNGD_MFCC = librosa.feature.delta(D_HNGD_MFCC, width=delta_win_size, axis=0)
    HNGDMFCC_D_DD = HNGDMFCC
    HNGDMFCC_D_DD = np.append(HNGDMFCC_D_DD, D_HNGD_MFCC, 1)
    HNGDMFCC_D_DD = np.append(HNGDMFCC_D_DD, DD_HNGD_MFCC, 1)


    delta_win_size = np.min([PARAMS['delta_win_size'], np.shape(MGDCC)[0]])
    if delta_win_size%2==0: # delta_win_size must be an odd integer >=3
        delta_win_size = np.max([delta_win_size-1, 3])
    if np.shape(MGDCC)[0]<3:
        MGDCC = np.append(MGDCC, MGDCC, axis=0)
    D_MGDCC = librosa.feature.delta(MGDCC, width=delta_win_size, axis=0)
    DD_MGDCC = librosa.feature.delta(D_MGDCC, width=delta_win_size, axis=0)
    MGDCC_D_DD = MGDCC
    MGDCC_D_DD = np.append(MGDCC_D_DD, D_MGDCC, 1)
    MGDCC_D_DD = np.append(MGDCC_D_DD, DD_MGDCC, 1)


    delta_win_size = np.min([PARAMS['delta_win_size'], np.shape(IFCC)[0]])
    if delta_win_size%2==0: # delta_win_size must be an odd integer >=3
        delta_win_size = np.max([delta_win_size-1, 3])
    if np.shape(IFCC)[0]<3:
        IFCC = np.append(IFCC, IFCC, axis=0)
    D_IFCC = librosa.feature.delta(IFCC, width=delta_win_size, axis=0)
    DD_IFCC = librosa.feature.delta(D_IFCC, width=delta_win_size, axis=0)
    IFCC_D_DD = IFCC
    IFCC_D_DD = np.append(IFCC_D_DD, D_IFCC, 1)
    IFCC_D_DD = np.append(IFCC_D_DD, DD_IFCC, 1)
    # IFCC_D_DD = IFCC
    
    return HNGDMFCC_D_DD.astype(np.float32), timeTaken_hngdmfcc, MGDCC_D_DD.astype(np.float32), timeTaken_mgdcc, IFCC_D_DD.astype(np.float32), timeTaken_ifcc




def compute_mgdcc_rho_gamma(PARAMS, Xin, fs):
    '''
    The input audio is split into 1sec segments to compute these features
    because the IFCC computation process takes an extremely long time for long audio
    sequences
    '''
    grp_phase = np.empty([])
    MGDCC = np.empty([])
    MGDCC_D_DD = np.empty([])

    frame_size = int(PARAMS['Tw']*fs/1000)
    frame_shift = int(PARAMS['Ts']*fs/1000)
    frames = librosa.util.frame(Xin, frame_size, frame_shift, axis=0)
    nFrames = np.shape(frames)[0]
    
    timeTaken_mgdcc = 0
    segment = 0
    segment_dur = 0

    frmStart = 0
    frmEnd = 0
    for frmStart in range(0, nFrames, PARAMS['intervalSize']):
        frmEnd = np.min([frmStart + PARAMS['intervalSize'], nFrames])
        frames_part = frames[frmStart:frmEnd, :]
        # print('Frames: ', np.shape(frames_part))
        
        startTime_mgdcc = time.time()
        grp_phase_part, MGDCC_part = mgd.compute_mgdcc(PARAMS, frames_part, fs, PARAMS['gamma'], PARAMS['rho'])
        if np.size(grp_phase)<=1:
            grp_phase = grp_phase_part
            MGDCC = MGDCC_part
        else:
            # print('grp_phase: ', np.shape(grp_phase), np.shape(grp_phase_part))
            grp_phase = np.append(grp_phase, grp_phase_part, axis=1)
            # print('MGDCC_D_DD: ', np.shape(MGDCC_D_DD), np.shape(MGDCC_D_DD_part))
            MGDCC = np.append(MGDCC, MGDCC_part, axis=0)
        endTime_mgdcc = time.time()
        timeTaken_mgdcc += endTime_mgdcc-startTime_mgdcc

        segment += 1
        segment_dur += (np.shape(frames_part)[0]/nFrames)*(len(Xin)/fs)
        print('\tSegment %.3d (%.2f sec/%.2f sec)\tTime taken: mgdcc=%.2f sec' % (segment+1, segment_dur, len(Xin)/fs, timeTaken_mgdcc), end='\r', flush=True)
    print(' ')

    delta_win_size = np.min([PARAMS['delta_win_size'], np.shape(MGDCC)[0]])
    if delta_win_size%2==0: # delta_win_size must be an odd integer >=3
        delta_win_size = np.max([delta_win_size-1, 3])
    if np.shape(MGDCC)[0]<3:
        MGDCC = np.append(MGDCC, MGDCC, axis=0)
    D_MGDCC = librosa.feature.delta(MGDCC, width=delta_win_size, axis=0)
    DD_MGDCC = librosa.feature.delta(D_MGDCC, width=delta_win_size, axis=0)
    MGDCC_D_DD = MGDCC
    MGDCC_D_DD = np.append(MGDCC_D_DD, D_MGDCC, 1)
    MGDCC_D_DD = np.append(MGDCC_D_DD, DD_MGDCC, 1)
    
    return MGDCC_D_DD.astype(np.float32), timeTaken_mgdcc
