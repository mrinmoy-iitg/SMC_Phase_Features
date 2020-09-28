#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:11:00 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import datetime
import os
import librosa.display
import glob
import lib.feature.preprocessing as preproc
import configparser
import lib.feature.khonglah_et_al as khonglah
import lib.feature.sell_et_al as sell
import lib.feature.phase_features as phfeat





def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Feature_Computation.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'data_folder': section['data_folder'],
            'dataset_name': section['dataset_name'],
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
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    print('\n\n\n', PARAMS['today'])
    
    opDir = PARAMS['output_folder'] + '/' + PARAMS['dataset_name'] + '/IFDur=' + str(PARAMS['intervalSize']) + 'frms_Tw=' + str(PARAMS['Tw']) + 'ms_Ts=' + str(PARAMS['Ts']).replace('.','-') + 'ms_' + PARAMS['today'] + '/'    
    
    if not os.path.exists(opDir):
        os.makedirs(opDir)
    
    for clNum in PARAMS['classes'].keys():
        fold = PARAMS['classes'][clNum]        
        path = PARAMS['data_folder'] + '/' + fold 	#path where the audio is
        files = librosa.util.find_files(path, ext=['wav'])
        numFiles = np.size(files)
        
        randFileIdx = list(range(np.size(files)))
        # random.shuffle(randFileIdx)
        count = 1	#to see in the terminal how many files have been loaded
        
        for f in range(numFiles): #numFiles
            audio = files[randFileIdx[f]]
            print('%s file (%.3d/%.3d)\t%s' % (fold, count, numFiles, audio.split('/')[-1]), end='\t', flush=True)
            
            
            '''
            Checking if feature file already exists ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            '''
            opDirFold_khonglah_et_al = opDir + '/Khonglah_et_al/' + fold + '/'
            if not os.path.exists(opDirFold_khonglah_et_al):
                os.makedirs(opDirFold_khonglah_et_al)
            fName_khonglah_et_al = opDirFold_khonglah_et_al + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
            matchingFiles = glob.glob(fName_khonglah_et_al)
            khonglah_et_al_file_exists = False
            if len(matchingFiles)>0:
                print('\n\t\t\tKhonglah et. al feature exists')
                khonglah_et_al_file_exists = True

            opDirFold_sell_et_al = opDir + '/Sell_et_al/' + fold + '/'
            if not os.path.exists(opDirFold_sell_et_al):
                os.makedirs(opDirFold_sell_et_al)
            fName_sell_et_al = opDirFold_sell_et_al + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
            matchingFiles = glob.glob(fName_sell_et_al)
            sell_et_al_file_exists = False
            if len(matchingFiles)>0:
                print('\t\t\tSell et. al feature exists')
                sell_et_al_file_exists = True

            opDirFold_mfcc = opDir + '/MFCC/' + fold + '/'
            if not os.path.exists(opDirFold_mfcc):
                os.makedirs(opDirFold_mfcc)
            fName_mfcc = opDirFold_mfcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
            matchingFiles = glob.glob(fName_mfcc)
            mfcc_file_exists = False
            if len(matchingFiles)>0:
                print('\t\t\tMFCC feature exists')
                mfcc_file_exists = True

            opDirFold_melspectrogram = opDir + '/Melspectrogram/' + fold + '/'
            if not os.path.exists(opDirFold_melspectrogram):
                os.makedirs(opDirFold_melspectrogram)
            fName_melspectrogram = opDirFold_melspectrogram + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
            matchingFiles = glob.glob(fName_melspectrogram)
            melspectrogram_file_exists = False
            if len(matchingFiles)>0:
                print('\t\t\tMelspectrogram feature exists')
                melspectrogram_file_exists = True

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
            phase_feat_file_exists = False
            if len(matchingFiles)>0:
                print('\t\t\tPhase features exist')
                phase_feat_file_exists = True
            
            if all([khonglah_et_al_file_exists, sell_et_al_file_exists, mfcc_file_exists, melspectrogram_file_exists, phase_feat_file_exists]):
                count += 1
                print('\n\n\n')
                continue
            '''
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            '''



            Xin, fs = librosa.core.load(audio, mono=True, sr=16000)
            
            if PARAMS['preemphasis']:
                Xin = librosa.effects.preemphasis(Xin)                                    
            
            Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin, fs=fs, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], alpha=PARAMS['silThresh'], beta=0.1)
            
            if np.size(Xin_silrem)<=0.1:
                continue
            print('File duration: ', np.round(len(Xin_silrem)/fs,2),' sec ( with silence=', np.round(len(Xin)/fs,2), ' sec)', end='\n', flush=True)
            Xin = Xin_silrem.copy()
            if len(Xin)/fs < 1:
                continue
            
            for PARAMS['featName'] in PARAMS['all_features']:
                
                '''
                Khonglah et al. features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='Khonglah_et_al':
                    if not khonglah_et_al_file_exists:
                        FV_khonglah_et_al = khonglah.compute_Khonglah_et_al_features(PARAMS, Xin, fs)
                        print('\tKhonglah et al. features computed: ', np.shape(FV_khonglah_et_al))
                        np.save(fName_khonglah_et_al, FV_khonglah_et_al)
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''


                '''
                Sell et al. features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='Sell_et_al':
                    if not sell_et_al_file_exists:
                        FV_sell_et_al = sell.compute_Sell_et_al_features(PARAMS, Xin, fs, frame_silMarker)
                        print('\tSell et al. features computed: ', np.shape(FV_sell_et_al))
                        np.save(fName_sell_et_al, FV_sell_et_al)
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
    
    
                '''
                MFCC features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='MFCC':
                    if not mfcc_file_exists:
                        Nframesize = int(PARAMS['Tw']*fs/1000)
                        Nframeshift = int(PARAMS['Ts']*fs/1000)
                        FV_mfcc = librosa.feature.mfcc(y=Xin, sr=fs, n_mfcc=PARAMS['n_mfcc'], n_fft=Nframesize, hop_length=Nframeshift, center=False, n_mels=PARAMS['n_mels'])
                        FV_mfcc = FV_mfcc.T
                        delta_win_size = np.min([PARAMS['delta_win_size'], np.shape(FV_mfcc)[0]])
                        if delta_win_size%2==0: # delta_win_size must be an odd integer >=3
                            delta_win_size = np.max([delta_win_size-1, 3])
                        if np.shape(FV_mfcc)[0]<3:
                            FV_mfcc = np.append(FV_mfcc, FV_mfcc, axis=0)
                        # Corrected: DeltaDelta was being computed as Delta
                        D_FV_mfcc = librosa.feature.delta(FV_mfcc, width=delta_win_size, axis=0)
                        DD_FV_mfcc = librosa.feature.delta(D_FV_mfcc, width=delta_win_size, axis=0)
                        FV_mfcc_D_DD = FV_mfcc
                        FV_mfcc_D_DD = np.append(FV_mfcc_D_DD, D_FV_mfcc, 1)
                        FV_mfcc_D_DD = np.append(FV_mfcc_D_DD, DD_FV_mfcc, 1)
                        FV_mfcc_D_DD = np.array(FV_mfcc_D_DD).astype(np.float32)
                        print('\tMFCC features computed: ', np.shape(FV_mfcc_D_DD))
                        np.save(fName_mfcc, FV_mfcc_D_DD)
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
    
    
                '''
                Melspectrogram features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
                if PARAMS['featName']=='Melspectrogram':
                    if not melspectrogram_file_exists:
                        Nframesize = int(PARAMS['Tw']*fs/1000)
                        Nframeshift = int(PARAMS['Ts']*fs/1000)
                        FV_melspectrogram = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, center=False, n_mels=PARAMS['n_mels'])
                        FV_melspectrogram = np.array(FV_melspectrogram.T).astype(np.float32)
                        print('\tMelspectrogram features computed: ', np.shape(FV_melspectrogram))
                        np.save(fName_melspectrogram, FV_melspectrogram)
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
    
                
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Phase-based features
                '''
                if PARAMS['featName']=='Phase_based':
                    if not phase_feat_file_exists:
                        HNGDMFCC_D_DD, timeTaken_hngdmfcc, MGDCC_D_DD, timeTaken_mgdcc, IFCC_D_DD, timeTaken_ifcc = phfeat.compute_phase_based_features(PARAMS, Xin, fs)            
                        np.save(fName_hngdmfcc, HNGDMFCC_D_DD)
                        print('\tHNGDMFCC feature computed: (%d, %d)\t%.2f secs' % (np.shape(HNGDMFCC_D_DD)[0], np.shape(HNGDMFCC_D_DD)[1], timeTaken_hngdmfcc))
            
                        np.save(fName_mgdcc, MGDCC_D_DD)
                        print('\tMGDCC feature computed: (%d, %d)\t%.2f secs' % (np.shape(MGDCC_D_DD)[0], np.shape(MGDCC_D_DD)[1], timeTaken_mgdcc))
        
                        np.save(fName_ifcc, IFCC_D_DD)
                        print('\tIFCC feature computed: (%d, %d)\t%.2f secs' % (np.shape(IFCC_D_DD)[0], np.shape(IFCC_D_DD)[1], timeTaken_ifcc), end='\n\n\n')
                '''
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                '''
    
            count += 1
            