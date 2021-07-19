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
import matplotlib.pyplot as plt
import lib.feature.phase_features as phfeat





def test_silence(Xin, Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration):
    print('totalSilDuration: ', totalSilDuration)
    plt.figure()
    plt.subplot(211)
    plt.plot(Xin)
    plt.plot(sample_silMarker)
    plt.subplot(212)
    plt.plot(Xin_silrem)
    plt.show()

    plt.figure()
    stft = np.log(np.abs(librosa.core.stft(Xin, n_fft=2*int(PARAMS['Tw']*fs/1000), hop_length=int(PARAMS['Ts']*fs/1000))))
    stft = stft[-1::-1,:]
    mask = np.zeros(np.shape(stft))
    for i in range(len(frame_silMarker)):
        if frame_silMarker[i]==0:
            mask[:,i] = np.max(stft)
    mask = np.subtract(stft, mask)
    plt.subplot(111)
    plt.imshow(mask)
    plt.show()
    


def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Feature_Computation.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'data_folder': section['data_folder'],
            'dataset_name': section['dataset_name'],
            'output_folder': section['output_folder'],
            'sampling_rate': int(section['sampling_rate']),
            'Tw': int(section['Tw']), # frame size in miliseconds
            'Ts': int(section['Ts']), # frame shift in miliseconds
            'n_cep': int(section['n_cep']), # Number of cepstral coefficients
            'no_filt': int(section['no_filt']), # Number of Mel filters
            'numBands': int(section['numBands']), # NUmber of Narrowband components for IFCC
            'delta_win_size': 9,
            'NBANDS': 18,
            'NORDER': 1000,
            'LPFFc': 28,
            'K': 36,
            'silThresh': float(section['silThresh']),
            'preemphasis': section.getboolean('preemphasis'),
            'intervalSize': int(section['intervalSize']), # interval size in miliseconds
            'intervalShift': int(section['intervalShift']), # interval shift in miliseconds
            'phase_feat_delta': section.getboolean('phase_feat_delta'),
            'classes': {0:'wav'}, # {0:'music', 1:'speech'}, # {2:'music+noise', 3:'noise', 4:'speech+music', 5:'speech+music+noise', 6:'speech+noise'}, # {0:'wav'}, #, # {'music', 'music+noise', 'noise', 'speech', 'speech+music', 'speech+music+noise', 'speech+noise'}
            'all_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC', 'Melspectrogram', 'Phase_based'], # 'Khonglah_et_al', 'Sell_et_al', 'MFCC', 'Melspectrogram', 'Phase_based', 'MGDCC'
            'featName': '',
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    print('\n\n\n', PARAMS['today'])
    
    opDir = PARAMS['output_folder'] + '/' + PARAMS['dataset_name'] + '/IFDur=' + str(PARAMS['intervalSize']) + 'frms_Tw=' + str(PARAMS['Tw']) + 'ms_Ts=' + str(PARAMS['Ts']).replace('.','-') + 'ms_' + PARAMS['today'] + '/'
    if PARAMS['sampling_rate']==0:
        opDir = PARAMS['output_folder'] + '/' + PARAMS['dataset_name'] + '/IFDur=' + str(PARAMS['intervalSize']) + 'frms_Tw=' + str(PARAMS['Tw']) + 'ms_Ts=' + str(PARAMS['Ts']).replace('.','-') + 'ms_Original_Sampling_Rate_' + PARAMS['today'] + '/'
                
    
    if not os.path.exists(opDir):
        os.makedirs(opDir)
    
    for clNum in PARAMS['classes'].keys():
        fold = PARAMS['classes'][clNum]        
        path = PARAMS['data_folder'] + '/' + fold 	#path where the audio is
        print('path: ', path)
        files = librosa.util.find_files(path, ext=['wav'])
        numFiles = np.size(files)
        print('Num files: ', numFiles, fold)
        
        count = 1	#to see in the terminal how many files have been loaded
        
        for audio in files: #numFiles
            print('%s file (%.3d/%.3d)\t%s' % (fold, count, len(files), audio.split('/')[-1]), end='\t', flush=True)
            
            '''
            Checking if feature file already exists ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            '''
            feature_flag_list = []
            khonglah_et_al_file_exists = False
            if 'Khonglah_et_al' in PARAMS['all_features']:
                opDirFold_khonglah_et_al = opDir + '/Khonglah_et_al/' + fold + '/'
                if not os.path.exists(opDirFold_khonglah_et_al):
                    os.makedirs(opDirFold_khonglah_et_al)
                fName_khonglah_et_al = opDirFold_khonglah_et_al + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                matchingFiles = glob.glob(fName_khonglah_et_al)
                if len(matchingFiles)>0:
                    print('\n\t\t\tKhonglah et. al feature exists')
                    khonglah_et_al_file_exists = True
                feature_flag_list.append(khonglah_et_al_file_exists)

            sell_et_al_file_exists = False
            if 'Sell_et_al' in PARAMS['all_features']:
                opDirFold_sell_et_al = opDir + '/Sell_et_al/' + fold + '/'
                if not os.path.exists(opDirFold_sell_et_al):
                    os.makedirs(opDirFold_sell_et_al)
                fName_sell_et_al = opDirFold_sell_et_al + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                matchingFiles = glob.glob(fName_sell_et_al)
                if len(matchingFiles)>0:
                    print('\t\t\tSell et. al feature exists')
                    sell_et_al_file_exists = True
                feature_flag_list.append(sell_et_al_file_exists)

            mfcc_file_exists = False
            if 'MFCC' in PARAMS['all_features']:
                opDirFold_mfcc = opDir + '/MFCC-' + str(3*PARAMS['n_cep']) + '/' + fold + '/'
                if not os.path.exists(opDirFold_mfcc):
                    os.makedirs(opDirFold_mfcc)
                fName_mfcc = opDirFold_mfcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                matchingFiles = glob.glob(fName_mfcc)
                if len(matchingFiles)>0:
                    print('\t\t\tMFCC-39 feature exists')
                    mfcc_file_exists = True
                feature_flag_list.append(mfcc_file_exists)

            melspectrogram_file_exists = False
            if 'Melspectrogram' in PARAMS['all_features']:
                opDirFold_melspectrogram = opDir + '/Melspectrogram/' + fold + '/'
                if not os.path.exists(opDirFold_melspectrogram):
                    os.makedirs(opDirFold_melspectrogram)
                fName_melspectrogram = opDirFold_melspectrogram + '/' + audio.split('/')[-1].split('.')[0] + '.npy'
                matchingFiles = glob.glob(fName_melspectrogram)
                if len(matchingFiles)>0:
                    print('\t\t\tMelspectrogram feature exists')
                    melspectrogram_file_exists = True
                feature_flag_list.append(melspectrogram_file_exists)

            phase_feat_file_exists = False
            if 'Phase_based' in PARAMS['all_features']:
                if PARAMS['phase_feat_delta']:
                    opDirFold_hngdmfcc = opDir + '/HNGDMFCC-' + str(3*PARAMS['n_cep']) + '/' + fold + '/'
                    opDirFold_mgdcc = opDir + '/MGDCC-' + str(3*PARAMS['n_cep']) + '/' + fold + '/'
                    opDirFold_ifcc = opDir + '/IFCC-' + str(3*PARAMS['n_cep']) + '/' + fold + '/'
                else:
                    opDirFold_hngdmfcc = opDir + '/HNGDMFCC_NoDelta/' + fold + '/'
                    opDirFold_mgdcc = opDir + '/MGDCC_NoDelta/' + fold + '/'
                    opDirFold_ifcc = opDir + '/IFCC_NoDelta/' + fold + '/'
                
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
                    print('\t\t\tPhase features exist')
                    phase_feat_file_exists = True
                feature_flag_list.append(phase_feat_file_exists)


            mgdcc_feat_file_exists = False
            if 'MGDCC' in PARAMS['all_features']:
                opDirFold_mgdcc = opDir + '/MGDCC-' + str(3*PARAMS['n_cep']) + '/' + fold + '/'                
                if not os.path.exists(opDirFold_mgdcc):
                    os.makedirs(opDirFold_mgdcc)
                fName_mgdcc = opDirFold_mgdcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'    
                # matchingFiles = glob.glob(fName_mgdcc)
                # if len(matchingFiles)>0:
                #     print('\t\t\tMGDCC feature exists')
                #     mgdcc_feat_file_exists = True
                if os.path.exists(fName_mgdcc):
                    print('\t\t\tMGDCC feature exists')
                    mgdcc_feat_file_exists = True
                feature_flag_list.append(mgdcc_feat_file_exists)
                    
                    
            
            # if all([khonglah_et_al_file_exists, sell_et_al_file_exists, mfcc_file_exists, melspectrogram_file_exists, phase_feat_file_exists, mgdcc_feat_file_exists]):
            #     count += 1
            #     print('\n\n\n')
            #     continue
            print('feature_flag_list: ', feature_flag_list)
            if all(feature_flag_list):
                count += 1
                print('\n\n\n')
                continue
            '''
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            '''
            
            if PARAMS['sampling_rate']>0:
                Xin, fs = librosa.core.load(audio, mono=True, sr=PARAMS['sampling_rate'])
            else:
                Xin, fs = librosa.core.load(audio, mono=True, sr=None)
                print('Original sampling rate: ', fs)
                
            
            if PARAMS['preemphasis']:
                Xin = librosa.effects.preemphasis(Xin)                                    
            
            Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin, fs=fs, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], alpha=PARAMS['silThresh'], beta=0.1)
            #test_silence(Xin, Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration)
            #continue
            
            # if np.size(Xin_silrem)<=1:
            #     continue
            print('File duration: ', np.round(len(Xin_silrem)/fs,2),' sec ( with silence=', np.round(len(Xin)/fs,2), ' sec)', end='\n', flush=True)
            # Xin = Xin_silrem.copy()
            # if len(Xin)/fs < 1:
            #     continue
            
            for PARAMS['featName'] in PARAMS['all_features']:
                
                ''' Khonglah et al. features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                if PARAMS['featName']=='Khonglah_et_al':
                    if not khonglah_et_al_file_exists:
                        FV_khonglah_et_al = khonglah.compute_Khonglah_et_al_features(PARAMS, Xin, fs)
                        print('\tKhonglah et al. features computed: ', np.shape(FV_khonglah_et_al))
                        np.save(fName_khonglah_et_al, FV_khonglah_et_al)
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''


                ''' Sell et al. features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                if PARAMS['featName']=='Sell_et_al':
                    if not sell_et_al_file_exists:
                        FV_sell_et_al = sell.compute_Sell_et_al_features(PARAMS, Xin, fs, frame_silMarker)
                        print('\tSell et al. features computed: ', np.shape(FV_sell_et_al))
                        np.save(fName_sell_et_al, FV_sell_et_al)
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
    
    
                ''' MFCC features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                if PARAMS['featName']=='MFCC':
                    if not mfcc_file_exists:
                        Nframesize = int(PARAMS['Tw']*fs/1000)
                        Nframeshift = int(PARAMS['Ts']*fs/1000)
                        FV_mfcc = librosa.feature.mfcc(y=Xin, sr=fs, n_mfcc=PARAMS['n_cep'], n_fft=Nframesize, hop_length=Nframeshift, center=False, n_mels=PARAMS['no_filt'])
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
                        print('\tMFCC-', str(3*PARAMS['n_cep']), ' features computed: ', np.shape(FV_mfcc_D_DD))
                        np.save(fName_mfcc, FV_mfcc_D_DD)
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
    
    
                ''' Melspectrogram features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                if PARAMS['featName']=='Melspectrogram':
                    if not melspectrogram_file_exists:
                        Nframesize = int(PARAMS['Tw']*fs/1000)
                        Nframeshift = int(PARAMS['Ts']*fs/1000)
                        FV_melspectrogram = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=Nframesize, hop_length=Nframeshift, center=False, n_mels=PARAMS['no_filt'])
                        FV_melspectrogram = np.array(FV_melspectrogram.T).astype(np.float32)
                        print('\tMelspectrogram features computed: ', np.shape(FV_melspectrogram))
                        np.save(fName_melspectrogram, FV_melspectrogram)
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
    
    
                
                ''' Phase-based features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                if PARAMS['featName']=='Phase_based':
                    if not phase_feat_file_exists:
                        if PARAMS['phase_feat_delta']:
                            HNGDMFCC_D_DD, timeTaken_hngdmfcc, MGDCC_D_DD, timeTaken_mgdcc, IFCC_D_DD, timeTaken_ifcc = phfeat.compute_phase_based_features(PARAMS, Xin, fs)            
                        else:
                            HNGDMFCC_D_DD, timeTaken_hngdmfcc, MGDCC_D_DD, timeTaken_mgdcc, IFCC_D_DD, timeTaken_ifcc = phfeat.compute_phase_based_features_nodelta(PARAMS, Xin, fs)            
                        np.save(fName_hngdmfcc, HNGDMFCC_D_DD)
                        print('\tHNGDMFCC-', str(3*PARAMS['n_cep']), ' feature computed: (%d, %d)\t%.2f secs' % (np.shape(HNGDMFCC_D_DD)[0], np.shape(HNGDMFCC_D_DD)[1], timeTaken_hngdmfcc))
            
                        np.save(fName_mgdcc, MGDCC_D_DD)
                        print('\tMGDCC-', str(3*PARAMS['n_cep']), ' feature computed: (%d, %d)\t%.2f secs' % (np.shape(MGDCC_D_DD)[0], np.shape(MGDCC_D_DD)[1], timeTaken_mgdcc))
        
                        np.save(fName_ifcc, IFCC_D_DD)
                        print('\tIFCC-', str(3*PARAMS['n_cep']), ' feature computed: (%d, %d)\t%.2f secs' % (np.shape(IFCC_D_DD)[0], np.shape(IFCC_D_DD)[1], timeTaken_ifcc), end='\n\n\n')
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''


                ''' MGDCC rho gamma features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
                if PARAMS['featName']=='MGDCC':
                    PARAMS['gamma'] = 0.3
                    PARAMS['rho'] = 0.1
                    if not mgdcc_feat_file_exists:
                        MGDCC_D_DD, timeTaken_mgdcc = phfeat.compute_mgdcc_rho_gamma(PARAMS, Xin, fs)            
                        np.save(fName_mgdcc, MGDCC_D_DD)
                        print('\tMGDCC-', str(3*PARAMS['n_cep']), ' feature computed: (%d, %d)\t%.2f secs' % (np.shape(MGDCC_D_DD)[0], np.shape(MGDCC_D_DD)[1], timeTaken_mgdcc), ' rho:', PARAMS['rho'], ' gamma:', PARAMS['gamma'])
                ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

    
            count += 1
            