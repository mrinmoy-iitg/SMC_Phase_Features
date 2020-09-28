#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:01:10 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import datetime
import os
import librosa.display
import random
import glob
import lib.feature.preprocessing as preproc
import configparser
import lib.feature.khonglah_et_al as khonglah
import lib.feature.sell_et_al as sell
import lib.feature.phase_features as phfeat




def normalize_signal(Xin):
    Xin = Xin - np.mean(Xin)
    Xin = Xin / (np.max(np.abs(Xin)) + 10e-8)
    return Xin
    



def signal_mixing(Xin_data, Xin_noise, target_dB):
    sp_len = len(Xin_data)
    mu_len = len(Xin_noise)
    common_len = np.min([sp_len, mu_len])
    if len(Xin_data)>common_len:
        len_diff = len(Xin_data)-common_len
        random_start_sample = np.random.randint(len_diff)
        Xin_data = Xin_data[random_start_sample:random_start_sample+common_len]
    else:
        Xin_data = Xin_data[:common_len]
        
    if len(Xin_noise)>common_len:
        len_diff = len(Xin_noise)-common_len
        random_start_sample = np.random.randint(len_diff)
        Xin_noise = Xin_noise[random_start_sample:random_start_sample+common_len]
    else:
        Xin_noise = Xin_noise[:common_len]
        
    data_energy = np.sum(np.power(Xin_data,2))/len(Xin_data)
    noise_energy = np.sum(np.power(Xin_noise,2))/len(Xin_noise)
    
    req_noise_energy = data_energy/np.power(10,(target_dB/10))
    mu_mult_fact = np.sqrt(req_noise_energy/noise_energy)
    Xin_noise_scaled = mu_mult_fact*Xin_noise
    
    Xin_mix = Xin_data + Xin_noise_scaled
    Xin_mix = normalize_signal(Xin_mix)
    
    return Xin_mix




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Feature_Computation_Noise.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'data_folder': section['data_folder'],
            'dataset_name': section['dataset_name'],
            'noise_folder': section['noise_folder'],
            'output_folder': section['output_folder'],
            'Tw': int(section['Tw']), # frame size in miliseconds
            'Ts': int(section['Ts']), # frame shift in miliseconds
            'n_mfcc': int(section['n_mfcc']),
            'n_mels': int(section['n_mels']),
            'no_filt': int(section['no_filt']),
            'n_cep': int(section['n_cep']),
            'numBands': int(section['numBands']),
            'silThresh': float(section['silThresh']),
            'preemphasis': section.getboolean('preemphasis'),
            'intervalSize': int(section['intervalSize']), # interval size in miliseconds
            'intervalShift': int(section['intervalShift']), # interval shift in miliseconds
            'L': 22,
            'delta_win_size': 9,
            'NBANDS': 18,
            'NORDER': 1000,
            'LPFFc': 28,
            'K': 36,
            'classes': {0:'music', 1:'speech'},
            'all_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC', 'Melspectrogram', 'Phase_based'],
            'featName': '',
            'noise_levels': [10, 8, 5, 2, 1, 0], # in dB
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    print('\n\n\n', PARAMS['today'])
    
    opDir = PARAMS['output_folder'] + '/' + PARAMS['dataset_name'] + '_Noisy/IFDur=' + str(PARAMS['intervalSize']) + 'frms_Tw=' + str(PARAMS['Tw']) + 'ms_Ts=' + str(PARAMS['Ts']).replace('.','-') + 'ms_' + PARAMS['today'] + '/'    
    
    if not os.path.exists(opDir):
        os.makedirs(opDir)

    noise_files = librosa.util.find_files(PARAMS['noise_folder'], ext=['wav'])
    
    for clNum in PARAMS['classes'].keys():
        fold = PARAMS['classes'][clNum]        
        path = PARAMS['data_folder'] + '/' + fold 	#path where the audio is
        files = librosa.util.find_files(path, ext=['wav'])
        numFiles = np.size(files)
        
        randFileIdx = list(range(np.size(files)))
        random.shuffle(randFileIdx)
        count = 1 # to see in the terminal how many files have been loaded
        
        for f in range(numFiles): #numFiles
            audio = files[randFileIdx[f]]
            print('%s file (%.3d/%.3d)\t%s' % (fold, count, numFiles, audio.split('/')[-1]), end='\t', flush=True)

            Xin_data, fs = librosa.core.load(audio, mono=True, sr=16000)
            if PARAMS['preemphasis']:
                Xin_data = librosa.effects.preemphasis(Xin_data)                                    
            
            Xin_data_silrem, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin_data, fs=fs, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], alpha=PARAMS['silThresh'], beta=0.1)            
            if np.size(Xin_data_silrem)<=1:
                continue
            print('File duration: ', np.round(len(Xin_data_silrem)/fs,2),' sec ( with silence=', np.round(len(Xin_data)/fs,2), ' sec)', end='\n', flush=True)
            Xin_data = Xin_data_silrem.copy()
            if len(Xin_data)/fs < 0.1:
                continue

            np.random.shuffle(noise_files)
            Xin_noise, fs = librosa.core.load(noise_files[0], mono=True, sr=16000)

            for PARAMS['featName'] in PARAMS['all_features']:
                '''
                Khonglah et al. features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='Khonglah_et_al':
                    opDirFold_khonglah_et_al = opDir + '/Khonglah_et_al/' + fold + '/'
                    if not os.path.exists(opDirFold_khonglah_et_al):
                        os.makedirs(opDirFold_khonglah_et_al)
                    fName_khonglah_et_al = opDirFold_khonglah_et_al + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                    matchingFiles = glob.glob(fName_khonglah_et_al)
                    if len(matchingFiles)>0:
                        print('File exists!!!\t',matchingFiles,'\n\n\n')
                    else:
                        FV_khonglah_et_al = {}
                        for target_dB in PARAMS['noise_levels']:
                            Xin_mix = signal_mixing(Xin_data, Xin_noise, target_dB)
                            FV_khonglah_et_al_db = None
                            FV_khonglah_et_al_db = khonglah.compute_Khonglah_et_al_features(PARAMS, Xin_mix, fs)
                            nFrames = np.shape(FV_khonglah_et_al_db)[0]
                            sparse_idx = []
                            for idx in range(0, nFrames, 1000):
                                sparse_idx.extend(list(range(idx, np.min([idx+68, nFrames]))))
                            sparse_idx = np.array(sparse_idx)
                            FV_khonglah_et_al[target_dB] = FV_khonglah_et_al_db[sparse_idx,:]
                            print('\tKhonglah et al. features computed: ', np.shape(FV_khonglah_et_al_db), np.shape(FV_khonglah_et_al[target_dB]), target_dB, 'dB')
                        np.save(fName_khonglah_et_al, FV_khonglah_et_al, allow_pickle=True)
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''


                '''
                Sell et al. features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='Sell_et_al':
                    opDirFold_sell_et_al = opDir + '/Sell_et_al/' + fold + '/'
                    if not os.path.exists(opDirFold_sell_et_al):
                        os.makedirs(opDirFold_sell_et_al)
                    fName_sell_et_al = opDirFold_sell_et_al + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                    matchingFiles = glob.glob(fName_sell_et_al)
                    if len(matchingFiles)>0:
                        print('File exists!!!\t',matchingFiles,'\n\n\n')
                    else:
                        FV_sell_et_al = {}
                        for target_dB in PARAMS['noise_levels']:
                            Xin_mix = signal_mixing(Xin_data, Xin_noise, target_dB)
                            FV_sell_et_al_db = None
                            FV_sell_et_al_db = sell.compute_Sell_et_al_features(PARAMS, Xin_mix, fs, frame_silMarker)
                            nFrames = np.shape(FV_sell_et_al_db)[0]
                            sparse_idx = []
                            for idx in range(0, nFrames, 1000):
                                sparse_idx.extend(list(range(idx, np.min([idx+68, nFrames]))))
                            sparse_idx = np.array(sparse_idx)
                            FV_sell_et_al[target_dB] = FV_sell_et_al_db[sparse_idx,:]
                            print('\tSell et al. features computed: ', np.shape(FV_sell_et_al_db), np.shape(FV_sell_et_al[target_dB]), target_dB, 'dB')
                        np.save(fName_sell_et_al, FV_sell_et_al, allow_pickle=True)
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
    
    
                '''
                MFCC features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='MFCC':
                    opDirFold_mfcc = opDir + '/MFCC/' + fold + '/'
                    if not os.path.exists(opDirFold_mfcc):
                        os.makedirs(opDirFold_mfcc)
                    fName_mfcc = opDirFold_mfcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                    matchingFiles = glob.glob(fName_mfcc)
                    if len(matchingFiles)>0:
                        print('File exists!!!\t',matchingFiles,'\n\n\n')
                    else:
                        Nframesize = int(PARAMS['Tw']*fs/1000)
                        Nframeshift = int(PARAMS['Ts']*fs/1000)
                        FV_mfcc_D_DD = {}
                        for target_dB in PARAMS['noise_levels']:
                            Xin_mix = signal_mixing(Xin_data, Xin_noise, target_dB)
                            FV_mfcc_db = None
                            FV_mfcc_db = librosa.feature.mfcc(y=Xin_mix, sr=fs, n_mfcc=PARAMS['n_mfcc'], n_fft=Nframesize, hop_length=Nframeshift, center=False, n_mels=PARAMS['n_mels'])
                            FV_mfcc_db = FV_mfcc_db.T
                            delta_win_size = np.min([PARAMS['delta_win_size'], np.shape(FV_mfcc_db)[0]])
                            if delta_win_size%2==0: # delta_win_size must be an odd integer >=3
                                delta_win_size = np.max([delta_win_size-1, 3])
                            if np.shape(FV_mfcc_db)[0]<3:
                                FV_mfcc_db = np.append(FV_mfcc_db, FV_mfcc_db, axis=0)
                            D_FV_mfcc_db = librosa.feature.delta(FV_mfcc_db, width=delta_win_size, axis=0)
                            DD_FV_mfcc_db = librosa.feature.delta(D_FV_mfcc_db, width=delta_win_size, axis=0)
                            FV_mfcc_D_DD_db = FV_mfcc_db
                            FV_mfcc_D_DD_db = np.append(FV_mfcc_D_DD_db, D_FV_mfcc_db, 1)
                            FV_mfcc_D_DD_db = np.append(FV_mfcc_D_DD_db, DD_FV_mfcc_db, 1)
                            FV_mfcc_D_DD_db = np.array(FV_mfcc_D_DD_db).astype(np.float32)

                            nFrames = np.shape(FV_mfcc_D_DD_db)[0]
                            sparse_idx = []
                            for idx in range(0, nFrames, 1000):
                                sparse_idx.extend(list(range(idx, np.min([idx+68, nFrames]))))
                            sparse_idx = np.array(sparse_idx)
                            FV_mfcc_D_DD[target_dB] = FV_mfcc_D_DD_db[sparse_idx,:]

                            print('\tMFCC features computed: ', np.shape(FV_mfcc_D_DD_db), np.shape(FV_mfcc_D_DD[target_dB]), target_dB, 'dB')
                        np.save(fName_mfcc, FV_mfcc_D_DD, allow_pickle=True)

                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
    
    
                '''
                Melspectrogram features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='Melspectrogram':
                    opDirFold_melspectrogram = opDir + '/Melspectrogram/' + fold + '/'
                    if not os.path.exists(opDirFold_melspectrogram):
                        os.makedirs(opDirFold_melspectrogram)
                    fName_melspectrogram = opDirFold_melspectrogram + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                    matchingFiles = glob.glob(fName_melspectrogram)
                    if len(matchingFiles)>0:
                        print('File exists!!!\t',matchingFiles,'\n\n\n')
                    else:
                        Nframesize = int(PARAMS['Tw']*fs/1000)
                        Nframeshift = int(PARAMS['Ts']*fs/1000)
                        FV_melspectrogram = {}
                        for target_dB in PARAMS['noise_levels']:
                            Xin_mix = signal_mixing(Xin_data, Xin_noise, target_dB)
                            FV_melspectrogram_db = None
                            FV_melspectrogram_db = librosa.feature.melspectrogram(y=Xin_mix, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, center=False, n_mels=PARAMS['n_mels'])
                            nFrames = np.shape(FV_melspectrogram_db)[1]
                            sparse_idx = []
                            for idx in range(0, nFrames, 1000):
                                sparse_idx.extend(list(range(idx, np.min([idx+68, nFrames]))))
                            sparse_idx = np.array(sparse_idx)
                            FV_melspectrogram[target_dB] = FV_melspectrogram_db[:, sparse_idx]
                            print('\tMelspectrogram features computed: ', np.shape(FV_melspectrogram_db), np.shape(FV_melspectrogram[target_dB]), target_dB, 'dB')
                        np.save(fName_melspectrogram, FV_melspectrogram, allow_pickle=True)

                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''

    
                    
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Phase-based features
                '''
                if PARAMS['featName']=='Phase_based':
                    opDirFold_hngdmfcc = opDir + '/HNGDMFCC/' + fold + '/'
                    opDirFold_mgdcc = opDir + '/MGDCC/' + fold + '/'
                    opDirFold_ifcc = opDir + '/IFCC/' + fold + '/'
                    
                    if not os.path.exists(opDirFold_hngdmfcc):
                        os.makedirs(opDirFold_hngdmfcc)
        
                    if not os.path.exists(opDirFold_mgdcc):
                        os.makedirs(opDirFold_mgdcc)
        
                    if not os.path.exists(opDirFold_ifcc):
                        os.makedirs(opDirFold_ifcc)

                    fName_hngdmfcc = opDirFold_hngdmfcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                    fName_mgdcc = opDirFold_mgdcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                    fName_ifcc = opDirFold_ifcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
        
                    matchingFiles = glob.glob(fName_hngdmfcc)
                    if len(matchingFiles)>0:
                        print('File exists!!!\t',matchingFiles,'\n\n\n')
                    else:
                        HNGDMFCC_D_DD = {}
                        MGDCC_D_DD = {}
                        IFCC_D_DD = {}
                        for target_dB in PARAMS['noise_levels']:
                            Xin_mix = signal_mixing(Xin_data, Xin_noise, target_dB)
                            HNGDMFCC_D_DD_db = None
                            MGDCC_D_DD_db = None
                            IFCC_D_DD_db = None
                            HNGDMFCC_D_DD_db, timeTaken_hngdmfcc, MGDCC_D_DD_db, timeTaken_mgdcc, IFCC_D_DD_db, timeTaken_ifcc = phfeat.compute_phase_based_features(PARAMS, Xin_mix, fs)            

                            nFrames = np.shape(HNGDMFCC_D_DD_db)[0]
                            sparse_idx = []
                            for idx in range(0, nFrames, 1000):
                                sparse_idx.extend(list(range(idx, np.min([idx+68, nFrames]))))
                            sparse_idx = np.array(sparse_idx)
                            HNGDMFCC_D_DD[target_dB] = HNGDMFCC_D_DD_db[sparse_idx,:]
                            MGDCC_D_DD[target_dB] = MGDCC_D_DD_db[sparse_idx, :]
                            IFCC_D_DD[target_dB] = IFCC_D_DD_db[sparse_idx, :]
                            print('\tHNGDMFCC feature computed: (%d, %d)\t (%d, %d)\t%.2f secs' % (np.shape(HNGDMFCC_D_DD_db)[0], np.shape(HNGDMFCC_D_DD_db)[1], np.shape(HNGDMFCC_D_DD[target_dB])[0], np.shape(HNGDMFCC_D_DD[target_dB])[1], timeTaken_hngdmfcc))
                            print('\tMGDCC feature computed: (%d, %d)\t (%d, %d)\t%.2f secs' % (np.shape(MGDCC_D_DD_db)[0], np.shape(MGDCC_D_DD_db)[1], np.shape(MGDCC_D_DD[target_dB])[0], np.shape(MGDCC_D_DD[target_dB])[1], timeTaken_mgdcc))
                            print('\tIFCC feature computed: (%d, %d)\t (%d, %d)\t%.2f secs' % (np.shape(IFCC_D_DD_db)[0], np.shape(IFCC_D_DD_db)[1], np.shape(IFCC_D_DD[target_dB])[0], np.shape(IFCC_D_DD[target_dB])[1], timeTaken_ifcc), target_dB, 'dB', end='\n\n\n')
            
                        np.save(fName_hngdmfcc, HNGDMFCC_D_DD, allow_pickle=True)
                        np.save(fName_mgdcc, MGDCC_D_DD, allow_pickle=True)
                        np.save(fName_ifcc, IFCC_D_DD, allow_pickle=True)

                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''

        
            count += 1
            