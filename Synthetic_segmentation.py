#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:57:40 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
import lib.classifier.cnn_classifier as CNN
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, medfilt
# from sklearn.feature_extraction.image import PatchExtractor
from lib.segmentation.cython_funcs import extract_patches as cextract_patches




def start_GPU_session():
    import tensorflow as tf
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1 , 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)



def reset_TF_session():
    import tensorflow as tf
    tf.compat.v1.keras.backend.clear_session()
    
    
    
def load_model(PARAMS, featName):
    PARAMS['modelName'] = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = PARAMS['modelName'].split('.')[0] + '.h5'
    architechtureFile = PARAMS['modelName'].split('.')[0] + '.json'
    paramFile = PARAMS['modelName'].split('.')[0] + '_params.npz'
    logFile = PARAMS['modelName'].split('.')[0] + '_log.csv'
    arch_file = PARAMS['modelName'].split('.')[0] + '_summary.txt'

    PARAMS['modelName'] = '.'.join(PARAMS['modelName'].split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    arch_file = '.'.join(arch_file.split('@'))
        
    epochs = np.load(paramFile)['epochs']
    batch_size = np.load(paramFile)['batch_size']
    input_shape = np.load(paramFile)['input_shape']
    learning_rate = np.load(paramFile)['lr']
    trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
    optimizer = optimizers.Adam(lr=learning_rate)
    
    try:
        with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
            model = model_from_json(f.read())
    except:
        model, learning_rate_temp = CNN.get_cnn_model(PARAMS['input_shape'][featName], 2)
    model.load_weights(weightFile) # Load weights into the new model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('CNN model exists! Loaded. Training time required=',trainingTimeTaken)
      
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'input_shape': input_shape,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def get_feature_patches(PARAMS, FV, patch_size, patch_shift, input_shape):
    # Removing NaN and Inf
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV[~np.isnan(FV).any(axis=1), :]
        FV = FV[~np.isinf(FV).any(axis=1), :]
    else:
        FV = FV[:, ~np.isnan(FV).any(axis=0)]
        FV = FV[:, ~np.isinf(FV).any(axis=0)]

    # FV should be of the shape (nFeatures, nFrames)
    # FV = StandardScaler(copy=False).fit_transform(FV)
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV.T
                
    patches = np.empty([])

    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)


    frmStart = 0
    frmEnd = 0
    # startTime = time.clock()
    for frmStart in range(0, np.shape(FV)[1], patch_shift):
        # print('get_feature_patches: ', frmStart, frmEnd, np.shape(FV))
        frmEnd = np.min([frmStart+patch_size, np.shape(FV)[1]])
        if frmEnd-frmStart<patch_size:
            frmStart = frmEnd - patch_size
        if np.size(patches)<=1:
            patches = np.array(FV[:, frmStart:frmEnd], ndmin=3)
        else:
            patches = np.append(patches, np.array(FV[:, frmStart:frmEnd], ndmin=3), axis=0)
    # print('My splitting: ', time.clock()-startTime, np.shape(patches))

    # numPatches = int(np.ceil(np.shape(FV)[1]/patch_shift))
    # patches = PatchExtractor(patch_size=(np.shape(FV)[0], patch_size), max_patches=numPatches).transform(np.expand_dims(FV, axis=0))
        
    if (np.shape(patches)[1]==9) or (np.shape(patches)[1]==10):
        diff_dim = input_shape[0]-np.shape(patches)[1]
        zero_padding = np.zeros((np.shape(patches)[0], diff_dim, np.shape(patches)[2]))
        patches = np.append(patches, zero_padding, axis=1)
    elif np.shape(patches)[1]==22:
        patches = patches[:,:21, :]
    elif np.shape(patches)[1]==39:
        if not PARAMS['39_dim_CC_feat']:
            first_7_cep_dim = np.array(list(range(0,7))+list(range(13,20))+list(range(26,33)))
            patches = patches[:, first_7_cep_dim, :]
    # print('Patches: ', np.shape(patches))
    
    return patches





def test_cnn_ensemble(PARAMS, Ensemble_Train_Params, file_sp, file_mu):
    count = -1
    # class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    temp_folder = PARAMS['opDir'] + '/__temp/fold' + str(PARAMS['fold']) + '/'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    count += 1
    PtdLabels = np.empty([])
    GroundTruth = np.empty([])
    Predictions = np.empty([])
    error = False
    print('\n\n\n')
    for featName in Ensemble_Train_Params.keys():
        Train_Params = Ensemble_Train_Params[featName]
        
        temp_file = 'pred_fold' + str(PARAMS['fold']) + '_' + featName + '_' + file_sp.split('.')[0] + '_' + file_mu.split('.')[0]
        print('Temp file: ', temp_file + '.pkl', os.path.exists(temp_folder + '/' + temp_file + '.pkl'))
        if not os.path.exists(temp_folder + '/' + temp_file + '.pkl'):
            fName_sp = PARAMS['test_folder'] + '/' + featName + '/speech/' + file_sp
            # print('fName_sp: ', fName_sp)
            if not os.path.exists(fName_sp):
                error = True
                break
            data_sp = np.load(fName_sp, allow_pickle=True)
            fName_mu = PARAMS['test_folder'] + '/' + featName + '/music/' + file_mu
            # print('fName_mu: ', fName_mu)
            if not os.path.exists(fName_mu):
                error = True
                break
            data_mu = np.load(fName_mu, allow_pickle=True)
            print('\t\t\t', featName, np.shape(data_sp), np.shape(data_mu))
            if np.shape(data_sp)[1]<np.shape(data_mu)[1]:
                data_mu = data_mu[:, :np.shape(data_sp)[1]]
            elif np.shape(data_sp)[1]>np.shape(data_mu)[1]:
                data_sp = data_sp[:, :np.shape(data_mu)[1]]
            data_sp = StandardScaler(copy=False).fit_transform(data_sp)
            data_mu = StandardScaler(copy=False).fit_transform(data_mu)
            data = np.append(data_sp, data_mu, axis=0)
            data = data.T
            print('\t\t\t data: ', np.shape(data.T))

            # batchData = get_feature_patches(PARAMS, data, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'][featName])
            
            batchData, label_sp, label_mu = cextract_patches(data, np.shape(data), PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], [1]*np.shape(data)[1], [0]*np.shape(data)[1], 'classification')
            if (np.shape(batchData)[1]==9) or (np.shape(batchData)[1]==10):
                diff_dim = PARAMS['input_shape'][featName][0]-np.shape(batchData)[1]
                zero_padding = np.zeros((np.shape(batchData)[0], diff_dim, np.shape(batchData)[2]))
                batchData = np.append(batchData, zero_padding, axis=1)
            elif np.shape(batchData)[1]==22:
                batchData = batchData[:,:21, :]
            elif np.shape(batchData)[1]==39:
                if not PARAMS['39_dim_CC_feat']:
                    first_7_cep_dim = np.array(list(range(0,7))+list(range(13,20))+list(range(26,33)))
                    batchData = batchData[:, first_7_cep_dim, :]
            # print('Patches: ', np.shape(patches))

            batchData = np.expand_dims(batchData, axis=3)        
            print('\t\t\t batchData: ', np.shape(batchData))
            
            pred = Train_Params['model'].predict(x=batchData)
            gt = np.ones(np.shape(batchData)[0])
            gt[np.shape(data_sp)[0]:] = 0
            misc.save_obj({'pred':pred, 'gt':gt}, temp_folder, temp_file)
        else:
            pred = misc.load_obj(temp_folder, temp_file)['pred']
            gt = misc.load_obj(temp_folder, temp_file)['gt']
        
        nPatches = len(gt)
        gt = gt[int(PARAMS['CNN_patch_size']/2):]
        gt = np.append(gt, [0]*(nPatches-len(gt)))
        print('nPatches: ', nPatches, len(gt))

        print('\t\t', featName, 'pred: ', np.shape(pred))
                
        if np.size(Predictions)<=1:
            Predictions = np.array(pred, ndmin=2)
            GroundTruth = gt
        else:
            print('\t\tPredictions: ', np.shape(Predictions), np.shape(pred))
            if np.shape(pred)[0]<np.shape(Predictions)[0]:
                while np.shape(pred)[0]<np.shape(Predictions)[0]:
                    d = np.shape(Predictions)[0]-np.shape(pred)[0]
                    pred = np.append(pred, np.array(pred[-d:,:], ndmin=2), axis=0)
            elif np.shape(pred)[0]>np.shape(Predictions)[0]:
                pred = pred[:np.shape(Predictions)[0], :]
            print('\t\tPredictions reshaped: ', np.shape(Predictions), np.shape(pred))
            Predictions = np.add(Predictions, np.array(pred, ndmin=2))
    
    print('\tPredictions scaling: ', np.shape(Predictions), np.shape(Ensemble_Train_Params))
    Predictions /= len(Ensemble_Train_Params)
    PtdLabels = np.argmax(Predictions, axis=1)
    if not error:    
        print('\t', np.shape(Predictions), ' acc=', np.round(np.sum(PtdLabels==GroundTruth)*100/np.size(GroundTruth), 2), end='\n')
    else:
        print('\tError!')

    return Predictions, PtdLabels, GroundTruth, error



def get_segment_level_statistics(ref_marker, est_marker, fold, feature_type, opFile):
    # print(np.shape(ref_marker), np.shape(est_marker))
    TP = np.sum(np.multiply((np.array(ref_marker)==1), (np.array(est_marker)==1)))
    FP = np.sum(np.multiply((np.array(ref_marker)==0), (np.array(est_marker)==1)))
    FN = np.sum(np.multiply((np.array(ref_marker)==1), (np.array(est_marker)==0)))
    TN = np.sum(np.multiply((np.array(ref_marker)==0), (np.array(est_marker)==0)))
    
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision_P = TP/(TP+FP)
    Recall_P = TP/(TP+FN)
    F1_P = 2*Precision_P*Recall_P/(Precision_P+Recall_P)

    Precision_N = TN/(TN+FN)
    Recall_N = TN/(TN+FP)
    F1_N = 2*Precision_N*Recall_N/(Precision_N+Recall_N)
    
    F1_avg = (F1_P+F1_N)/2

    results = {}
    results['0'] = 'Fold:'+str(fold)
    results['1'] = 'Feature_type:'+feature_type
    results['2'] = 'Accuracy:'+str(np.round(Accuracy, 4))
    results['3'] = 'Precision_sp'+':'+str(np.round(Precision_P, 4))
    results['4'] = 'Precision_mu'+':'+str(np.round(Precision_N, 4))
    results['5'] = 'Recall_sp'+':'+str(np.round(Recall_P, 4))
    results['6'] = 'Recall_mu'+':'+str(np.round(Recall_N, 4))
    results['7'] = 'F-measure_sp'+':'+str(np.round(F1_P, 4))
    results['8'] = 'F-measure_mu'+':'+str(np.round(F1_N, 4))
    results['9'] = 'F-measure_avg'+':'+str(np.round(F1_avg, 4))
    misc.print_analysis(opFile, results)
    
    print('Segment level statistics:\nA=%.2f, Prec(+ve)=%.4f, Rec(+ve)=%.4f, F1(+ve)=%.4f, Prec(-ve)=%.4f, Rec(-ve)=%.4f, F1(-ve)=%.4f, F1(avg)=%.4f\n' % (Accuracy, Precision_P, Recall_P, F1_P, Precision_N, Recall_N, F1_N, F1_avg))




def get_basic_statistics(twin_frms, ref_marker, est_marker):
    TP = 0 # True Positives
    FP = 0 # False Positives
    FN = 0 # False Negatives

    est_event_frms = np.squeeze(np.where(est_marker>0))
    ref_event_frms = np.squeeze(np.where(ref_marker>0))
    if np.size(est_event_frms)==1:
        est_event_frms = [est_event_frms]
    if np.size(ref_event_frms)==1:
        ref_event_frms = [ref_event_frms]
    
    if (np.size(est_event_frms)<1):
        FN = np.size(ref_event_frms)
        return TP, FP, FN
        
    if (np.size(ref_event_frms)<1):
        FP = np.size(est_event_frms)
        return TP, FP, FN

    ref_event_frms_count = np.zeros(len(ref_event_frms))
    if np.size(est_event_frms)>0:
        for i in range(len(est_event_frms)):
            TP_flag = 0
            for j in range(len(ref_event_frms)):
                if  np.abs(est_event_frms[i]-ref_event_frms[j])<=twin_frms:
                    TP_flag = j+1
                    ref_event_frms_count[j] += 1
                    break
            if (TP_flag>0):
                if ref_event_frms_count[TP_flag-1]==1:
                    TP += 1
                else:
                    FP += 1
            else:
                FP += 1
    FN = len(ref_event_frms)-TP # np.sum(ref_event_frms_count==0)

    return TP, FP, FN




def get_final_statistics(TPc, FPc, FNc, Nc, opFile, class_name, fold, feature_type, twin):
    Pc = TPc/(TPc+FPc+1e-10) # Precision
    Rc = TPc/(TPc+FNc+1e-10) # Recall
    Fc = 2*Pc*Rc/(Pc+Rc+1e-10) # F_measure
    Dc = FNc/(Nc+1e-10) # Deletion rate
    Ic = FPc/(Nc+1e-10) # Insertion rate
    Ec = Dc+Ic # Error rate

    results = {}
    results['0'] = 'Class:'+class_name
    results['1'] = 'Fold:'+str(fold)
    results['2'] = 'Feature type:'+ feature_type
    results['3'] = 'Tolerance:'+str(twin)+' ms'
    results['4'] = 'True Positives:'+str(TPc)
    results['5'] = 'False Positives:'+str(FPc)
    results['6'] = 'False Negatives:'+str(FNc)
    results['7'] = 'Precision:'+str(np.round(Pc, 4))
    results['8'] = 'Recall:'+str(np.round(Rc, 4))
    results['9'] = 'F-measure:'+str(np.round(Fc, 4))
    results['10'] = 'Deletion Rate:'+str(np.round(Dc, 4))
    results['11'] = 'Insertion Rate:'+str(np.round(Ic, 4))
    results['12'] = 'Error Rate:'+str(np.round(Ec, 4))
    misc.print_analysis(opFile, results)
    
    return Pc, Rc, Fc, Dc, Ic, Ec




def compute_segmentation_performance(PARAMS, GroundTruth, Frame_Predictions, tol_windows, feature_type, win_size=5, plot_fig=False):
    files = [fName for fName in Frame_Predictions.keys()]
    opDirFig = PARAMS['opDir'] + '/__figures/fold' + str(PARAMS['fold']) + '/' + feature_type + '/'
    if not os.path.exists(opDirFig):
        os.makedirs(opDirFig)
    Stats = {win:{'TP_sp':0, 'FP_sp':0, 'FN_sp':0, 'N_sp':0, 'TP_mu':0, 'FP_mu':0, 'FN_mu':0, 'N_mu':0} for win in tol_windows}
    for fl in files:

        if plot_fig:
            plt.figure()
            plt.subplot(311)
            plt.plot(GroundTruth[fl])
            plt.subplot(312)
            plt.plot(Frame_Predictions[fl][:,0])
            plt.plot(np.argmax(Frame_Predictions[fl], axis=1))

        if win_size>0:
            Frame_Predictions[fl][:,0] = medfilt(Frame_Predictions[fl][:,0], win_size)
            Frame_Predictions[fl][:,1] = 1-Frame_Predictions[fl][:,0]

        if plot_fig:
            plt.subplot(313)
            plt.plot(Frame_Predictions[fl][:,0])
            plt.plot(np.argmax(Frame_Predictions[fl], axis=1))
            plt.savefig(opDirFig+'/'+str(fl)+'.jpg', bbox_inches='tight')
            plt.close()
        
        PtdLabels_sp = np.argmax(Frame_Predictions[fl], axis=1)
        PtdLabels_mu = np.argmin(Frame_Predictions[fl], axis=1)

        ref_speech_marker = GroundTruth[fl]
        ref_music_marker = 1 - GroundTruth[fl]
        nFrames = len(ref_speech_marker)

        ref_speech_marker_diff = (np.abs(np.diff(ref_speech_marker))>0).astype(int)
        ref_music_marker_diff = (np.abs(np.diff(ref_music_marker))>0).astype(int)

        est_speech_marker = (np.abs(np.diff(PtdLabels_sp))>0).astype(int)
        est_music_marker = (np.abs(np.diff(PtdLabels_mu))>0).astype(int)

        for twin in tol_windows:
            twin_frms = int(np.ceil((twin/((nFrames*10)+15))*nFrames))
            
            TP_sp, FP_sp, FN_sp = get_basic_statistics(twin_frms, ref_speech_marker_diff, est_speech_marker)
            Stats[twin]['TP_sp'] += TP_sp
            Stats[twin]['FP_sp'] += FP_sp
            Stats[twin]['FN_sp'] += FN_sp
            Stats[twin]['N_sp'] += np.sum(ref_speech_marker_diff)
            print('Basic stats sp: ', np.sum(est_speech_marker), TP_sp, FP_sp, FN_sp)

            TP_mu, FP_mu, FN_mu = get_basic_statistics(twin_frms, ref_music_marker_diff, est_music_marker)
            Stats[twin]['TP_mu'] += TP_mu
            Stats[twin]['FP_mu'] += FP_mu
            Stats[twin]['FN_mu'] += FN_mu
            Stats[twin]['N_mu'] += np.sum(ref_music_marker_diff)
            print('Basic stats mu: ', np.sum(est_music_marker), TP_mu, FP_mu, FN_mu)
            
            prec_sp = TP_sp/(TP_sp+FP_sp+1e-10)
            rec_sp = TP_sp/(TP_sp+FN_sp+1e-10)
            fscore_sp = 2*prec_sp*rec_sp/(prec_sp+rec_sp+1e-10)
            prec_mu = TP_mu/(TP_mu+FP_mu+1e-10)
            rec_mu = TP_mu/(TP_mu+FN_mu+1e-10)
            fscore_mu = 2*prec_mu*rec_mu/(prec_mu+rec_mu+1e-10)
            # print(twin, fl, '\tFscore: ', prec_sp, rec_sp, fscore_sp, prec_mu, rec_mu, fscore_mu)
            print('Win=(%d,%d,%d)\t(P_sp:%.4f, R_sp:%.4f, F1_sp:%.4f)\t(P_mu:%.4f, R_mu:%.4f, F1_mu:%.4f): ' % (twin, twin_frms, nFrames, prec_sp, rec_sp, fscore_sp, prec_mu, rec_mu, fscore_mu))
    print('Stats: ', Stats)
        
    opFile = PARAMS['opDir'] + '/Event_Level_Performance.csv'
    for twin in tol_windows:
        
        # P_sp, R_sp, F_sp, D_sp, I_sp, E_sp = get_final_statistics(Stats['TP_sp'], Stats['FP_sp'], Stats['FN_sp'], Stats['N_sp'], opFile, 'speech', PARAMS['fold'], feature_type, twin)
        # P_mu, R_mu, F_mu, D_mu, I_mu, E_mu = get_final_statistics(Stats['TP_mu'], Stats['FP_mu'], Stats['FN_mu'], Stats['N_mu'], opFile, 'music', PARAMS['fold'], feature_type, twin)
        P, R, F, D, I, E = get_final_statistics(Stats[twin]['TP_sp']+Stats[twin]['TP_mu'], Stats[twin]['FP_sp']+Stats[twin]['FP_mu'], Stats[twin]['FN_sp']+Stats[twin]['FN_mu'], Stats[twin]['N_sp']+Stats[twin]['N_mu'], opFile, 'overall', PARAMS['fold'], feature_type, twin)




def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': '/scratch/mbhattacharjee/features/SMC_SPECOM/musan/IFDur=100frms_Tw=25ms_Ts=10ms_2020-08-09/',
            'test_path': '/scratch/mbhattacharjee/features/SMC_SPECOM/musan/IFDur=100frms_Tw=25ms_Ts=10ms_2020-08-09/',
            'dataset_name': 'MUSAN',
            'CNN_patch_size': 68,
            'CNN_patch_shift': 68,
            'CNN_patch_shift_test': 1,
            'CV_folds': 3,
            'save_flag': True,
            'use_GPU': True,
            'scale_data': True,
            'fold':0,
            'CNN_feat_dim': 0,
            'GPU_session':None,
            'output_folder':'',
            'test_folder': '',
            'opDir':'',
            'classes':{0:'music', 1:'speech'},
            # 'all_featName': ['MFCC-39'], 
            'all_featName': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'], # 'MGDCC-39_row_gamma'
            '39_dim_CC_feat': True,
            'featName':'',
            'modelName':'',
            'feat_combinations':{
                # 'Khonglah': ['Khonglah_et_al'],
                # 'Sell': ['Sell_et_al'],
                # 'MFCC': ['MFCC-39'],
                # 'Melspectrogram': ['Melspectrogram'],
                # 'HNGDMFCC': ['HNGDMFCC'],
                # 'MGDCC': ['MGDCC'],
                # 'IFCC': ['IFCC'],
                'all_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'],
                'magnitude_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram'],
                'phase_features': ['HNGDMFCC', 'MGDCC', 'IFCC'],
                'all_features-IFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC'],
                # 'magnitude_features+IFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'IFCC'],
                # 'all_features-MGDCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'IFCC'],
                # 'all_features-HNGDMFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'IFCC', 'MGDCC'],
                # 'magnitude_features+HNGDMFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC'],
                # 'magnitude_features+MGDCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'MGDCC'],
                }, #['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'HNGDMFCC', 'MGDCC', 'IFCC', 'Melspectrogram']
            }

    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    
    if PARAMS['test_path']=='':
        cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
        cv_file_list_test = cv_file_list
        PARAMS['test_folder'] = PARAMS['folder']        
    else:
        PARAMS['test_folder'] = PARAMS['test_path']
        cv_file_list = misc.create_CV_folds(PARAMS['test_folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
        cv_file_list_test = misc.create_CV_folds(PARAMS['test_folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])

    PARAMS['opDir'] = PARAMS['test_folder'] + '/__RESULTS/' + PARAMS['today'] + '/Synthetic_Segmentation/'
        
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
    
    misc.print_configuration(PARAMS)
    
    for foldNum in range(PARAMS['CV_folds']):
        PARAMS['fold'] = foldNum
        PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
        
        PARAMS['modelName'] = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/CNN/fold' + str(PARAMS['fold']) + '_model.xyz'

        if PARAMS['use_GPU']:
            start_GPU_session()

        PARAMS['input_shape'] = {}
        for featName in PARAMS['all_featName']:
            if featName in ['HNGDMFCC', 'MGDCC', 'IFCC', 'MFCC-39']:
                PARAMS['input_shape'][featName] = (39, PARAMS['CNN_patch_size'], 1)
            else:
                PARAMS['input_shape'][featName] = (21, PARAMS['CNN_patch_size'], 1)
        print('input_shape: ', PARAMS['input_shape'])

        for feature_type in PARAMS['feat_combinations'].keys():
            feature_list = PARAMS['feat_combinations'][feature_type]
            print('\n\n\nfeature_type: ', feature_type)
            print('feature_list: ', feature_list)
                    
            Ensemble_Train_Params = {}
            for featName in feature_list:
                PARAMS['modelName'] = PARAMS['folder'] + '/' + featName + '/__RESULTS/CNN/fold' + str(PARAMS['fold']) + '_model.xyz'
                print('modelName: ', PARAMS['modelName'])
                Train_Params = load_model(PARAMS, featName)
                Ensemble_Train_Params[featName] = Train_Params
            
            files_sp = PARAMS['test_files']['speech']
            # np.random.shuffle(files_sp)
            files_mu = PARAMS['test_files']['music']
            # np.random.shuffle(files_mu)
            nFiles = np.min([len(files_sp), len(files_mu)])
            
            All_Predictions = {}
            All_Labels = {}
            All_PtdLabels = []
            All_GroundTruths = []
            for i in range(nFiles):
                Predictions, PtdLabels, GroundTruths, error = test_cnn_ensemble(PARAMS, Ensemble_Train_Params, files_sp[i], files_mu[i])
                if error:
                    continue
                All_Predictions[i] = Predictions
                All_Labels[i] = GroundTruths
                
                plot_fig = False
                if plot_fig:
                    plt.figure()
                    plt.subplot(311)
                    plt.plot(GroundTruths)
                    plt.subplot(312)
                    plt.plot(PtdLabels)
                    plt.subplot(313)
                    plt.plot(Predictions[:,1])
                    plt.show()
                
                All_PtdLabels.extend(PtdLabels)
                All_GroundTruths.extend(GroundTruths)
                Predictions = None
                PtdLabels = None
                GroundTruths = None
                del Predictions
                del PtdLabels
                del GroundTruths
            
            # plt.subplot(211)
            # plt.plot(All_GroundTruths)
            # plt.subplot(212)
            # plt.plot(All_PtdLabels)
            # plt.show()
            
            ConfMat, fscore = misc.getPerformance(All_PtdLabels, All_GroundTruths)
            print('ConfMat: ', ConfMat)
            print('fscore: ', fscore)
            
            get_segment_level_statistics(All_GroundTruths, All_PtdLabels, PARAMS['fold'], feature_type, PARAMS['opDir']+'/Segment_Level_Performance.csv')
            
            compute_segmentation_performance(PARAMS, All_Labels, All_Predictions, [1000,500], feature_type, win_size=101, plot_fig=True)
        
        if PARAMS['use_GPU']:
            reset_TF_session()
            
