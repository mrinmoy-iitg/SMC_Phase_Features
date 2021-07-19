#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:09:28 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import datetime
import os
import librosa.display
import glob
import lib.feature.preprocessing as preproc
import configparser
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
            'classes': {0:'music', 1:'speech'}, # {2:'music+noise', 3:'noise', 4:'speech+music', 5:'speech+music+noise', 6:'speech+noise'}, # {0:'wav'}, #, # {'music', 'music+noise', 'noise', 'speech', 'speech+music', 'speech+music+noise', 'speech+noise'}
            'all_features': ['MGDCC_rho_gamma'], # 'Khonglah_et_al', 'Sell_et_al', 'MFCC', 'Melspectrogram', 'Phase_based', 'MGDCC_rho_gamma'
            'featName': '',
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    print('\n\n\n', PARAMS['today'])
    
    opDir = PARAMS['output_folder'] + '/' + PARAMS['dataset_name'] + '/IFDur=' + str(PARAMS['intervalSize']) + 'frms_Tw=' + str(PARAMS['Tw']) + 'ms_Ts=' + str(PARAMS['Ts']).replace('.','-') + 'ms_MGDCC_rho_gamma' + PARAMS['today'] + '/'
    if not os.path.exists(opDir):
        os.makedirs(opDir)
    for PARAMS['rho'] in np.arange(0.1,1.01,0.1):
        PARAMS['rho'] = np.round(PARAMS['rho'],1)
        for PARAMS['gamma'] in np.arange(0.1,1.01,0.1):
            PARAMS['gamma'] = np.round(PARAMS['gamma'],1)
    
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
                    mgdcc_feat_file_exists = False
                    opDirFold_mgdcc = opDir + '/MGDCC-' + str(3*PARAMS['n_cep']) + '_rho' + str(PARAMS['rho']) + '_gamma' + str(PARAMS['gamma']) + '/' + fold + '/'                
                    if not os.path.exists(opDirFold_mgdcc):
                        os.makedirs(opDirFold_mgdcc)
                    fName_mgdcc = opDirFold_mgdcc + '/' + audio.split('/')[-1].split('.')[0] + '.npy'    
                    matchingFiles = glob.glob(fName_mgdcc)
                    if len(matchingFiles)>0:
                        print('\t\t\tMGDCC feature exists')
                        mgdcc_feat_file_exists = True
                            
                    if mgdcc_feat_file_exists:
                        count += 1
                        print('\n\n\n')
                        continue
                    '''
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    '''
                    
                    Xin, fs = librosa.core.load(audio, mono=True, sr=PARAMS['sampling_rate'])
        
                    if PARAMS['preemphasis']:
                        Xin = librosa.effects.preemphasis(Xin)                                    
                    
                    Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin, fs=fs, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], alpha=PARAMS['silThresh'], beta=0.1)
                    #test_silence(Xin, Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration)
                    #continue
                    
                    if np.size(Xin_silrem)<=1:
                        continue
                    print('File duration: ', np.round(len(Xin_silrem)/fs,2),' sec ( with silence=', np.round(len(Xin)/fs,2), ' sec)', end='\n', flush=True)
                    Xin = Xin_silrem.copy()
                    if len(Xin)/fs < 1:
                        continue
                                    
                    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    MGDCC rho gamma features
                    '''
                    if not mgdcc_feat_file_exists:
                        MGDCC_D_DD, timeTaken_mgdcc = phfeat.compute_mgdcc_rho_gamma(PARAMS, Xin, fs)            
                        np.save(fName_mgdcc, MGDCC_D_DD)
                        print('\tMGDCC-', str(3*PARAMS['n_cep']), ' feature computed: (%d, %d)\t%.2f secs' % (np.shape(MGDCC_D_DD)[0], np.shape(MGDCC_D_DD)[1], timeTaken_mgdcc), ' rho:', PARAMS['rho'], ' gamma:', PARAMS['gamma'])
                    '''
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    '''
            
                    count += 1
                    