#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:33:02 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import librosa
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import lib.misc as misc
import os
from dictances import bhattacharyya
from scipy.signal import medfilt



def mode_filtering(X, win_size):
    if win_size%2==0:
        win_size += 1
    X_smooth = X.copy()
    for i in range(int(win_size/2), len(X)-int(win_size/2)):
        win = X[i-int(win_size/2):i+int(win_size/2)]
        uniq_lab, uniq_counts = np.unique(win, return_counts=True)
        X_smooth[i] = uniq_lab[np.argmax(uniq_counts)]    
    return X_smooth





def get_transition_track(X, win_size=3):
    # X[:,0] = savgol_filter(X[:,0], 3, polyorder=2)
    # X[:,1] = 1-X[:,0]
    half_win_size = int(win_size/2)
    bhattacharya_dist = np.zeros(np.shape(X)[0])
    for i in range(half_win_size,np.shape(X)[0]-half_win_size):
        left_hist = {
            'negative_mn':np.mean(X[i-half_win_size:i,0]), 
            # 'negative_sd':np.std(X[i-half_win_size:i,0]), 
            'positive_mn':np.mean(X[i-half_win_size:i,1]),
            # 'positive_sd':np.std(X[i-half_win_size:i,1]),
            }
        right_hist = {
            'negative_mn':np.mean(X[i+1:i+half_win_size+1,0]), 
            # 'negative_sd':np.std(X[i+1:i+half_win_size+1,0]), 
            'positive_mn':np.mean(X[i+1:i+half_win_size+1,1]),
            # 'positive_sd':np.std(X[i+1:i+half_win_size+1,1]),
            }
        bhattacharya_dist[i] = bhattacharyya(left_hist, right_hist)
        
    # bhattacharya_dist[np.squeeze(np.where(bhattacharya_dist==0))] = np.min(bhattacharya_dist)
    bhattacharya_dist[np.squeeze(np.where(bhattacharya_dist>(np.mean(bhattacharya_dist)+5*np.std(bhattacharya_dist))))] = np.min(bhattacharya_dist)
    # bhattacharya_dist = savgol_filter(bhattacharya_dist, win_size, polyorder=2)
    
    return bhattacharya_dist



def draw_figures(opDir, feature_type, fold, segmentation_track, fl, MU_MARKER, CNN_PtdLabels_mu, LSTM_PtdLabels_mu, CNN_Frame_Predictions, SP_MARKER, CNN_PtdLabels_sp, LSTM_PtdLabels_sp, smoothing_win_size):
    if segmentation_track:
        BD_win_size = smoothing_win_size
        Transition_Track_mu = get_transition_track(CNN_Frame_Predictions['Pred'], BD_win_size)
        Transition_Track_sp = get_transition_track(CNN_Frame_Predictions['Pred'], BD_win_size)

    # plt.figure()
    opDirFig = opDir + '/__segmentation_results/' + feature_type + '/figures/fold' + str(fold) + '/'
    if not os.path.exists(opDirFig):
        os.makedirs(opDirFig)
    
    if not segmentation_track:
        plt.title(fl.split('/')[0])

        plt.subplot(311)
        plt.plot(MU_MARKER, 'b-')
        plt.legend(['Music GroundTruth'])
        
        plt.subplot(312)
        plt.plot(CNN_PtdLabels_mu, 'g-')
        plt.legend(['CNN labels mu'])

        plt.subplot(313)
        plt.plot(LSTM_PtdLabels_mu, 'm-')
        plt.legend(['LSTM music labels'])

    else:
        plt.subplot(411)
        plt.plot(MU_MARKER, 'b-')
        plt.legend(['Music GroundTruth'])
        plt.title('Window size='+str(BD_win_size))    

        plt.subplot(412)
        X = CNN_Frame_Predictions['Pred'][:,1]
        # X = savgol_filter(X, 5, polyorder=3)
        plt.plot(X, 'g-')
        plt.legend(['CNN predictions mu'])

        plt.subplot(413)
        plt.plot(Transition_Track_mu, 'm-')
        plt.legend(['Bhattacharya dist'])

        plt.subplot(414)
        plt.plot(np.diff(Transition_Track_mu), 'm-')
        plt.legend(['Differenced Bhattacharya dist'])

    # plt.show()
    plt.savefig(opDirFig+fl.split('/')[-1].split('.')[0]+'_mu.jpg', bbox_inches='tight')
    plt.cla()

    # plt.figure()
    
    if not segmentation_track:
        plt.title(fl.split('/')[0])

        plt.subplot(311)
        plt.plot(SP_MARKER, 'b-')
        plt.legend(['Speech GroundTruth'])
        
        plt.subplot(312)
        plt.plot(CNN_PtdLabels_sp, 'g-')
        plt.legend(['CNN labels sp'])

        plt.subplot(313)
        plt.plot(LSTM_PtdLabels_sp, 'm-')
        plt.legend(['LSTM speech labels'])
        
    else:
        plt.subplot(411)
        plt.plot(SP_MARKER, 'b-')
        plt.legend(['Speech GroundTruth'])
        plt.title('Window size='+str(BD_win_size))    

        plt.subplot(412)
        X = CNN_Frame_Predictions['Pred'][:,1]
        # X = savgol_filter(X, 5, polyorder=3)
        plt.plot(X, 'g-')
        plt.legend(['CNN predictions sp'])

        plt.subplot(413)
        plt.plot(Transition_Track_sp, 'm-')
        plt.legend(['Bhattacharya dist'])

        plt.subplot(414)
        plt.plot(np.diff(Transition_Track_sp), 'm-')
        plt.legend(['Differenced Bhattacharya dist'])

    # plt.show()
    plt.savefig(opDirFig+fl.split('/')[-1].split('.')[0]+'_sp.jpg', bbox_inches='tight')
    plt.cla()

    


def get_segment_level_statistics(ref_marker, est_marker, class_name, fold, opFile, feature_type):
    # print(np.shape(ref_marker), np.shape(est_marker))
    TP = np.sum(np.multiply((np.array(ref_marker)==1), (np.array(est_marker)==1)))
    FP = np.sum(np.multiply((np.array(ref_marker)==0), (np.array(est_marker)==1)))
    FN = np.sum(np.multiply((np.array(ref_marker)==1), (np.array(est_marker)==0)))
    TN = np.sum(np.multiply((np.array(ref_marker)==0), (np.array(est_marker)==0)))
    print('segment_level_statistics:')
    print(TP, FN)
    print(FP, TN)
    
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision_P = TP/(TP+FP)
    Recall_P = TP/(TP+FN)
    F1_P = 2*Precision_P*Recall_P/(Precision_P+Recall_P)

    Precision_N = TN/(TN+FN)
    Recall_N = TN/(TN+FP)
    F1_N = 2*Precision_N*Recall_N/(Precision_N+Recall_N)

    results = {}
    results['0'] = 'Class:'+class_name
    results['1'] = 'Fold:'+str(fold)
    results['2'] = 'Feature type:'+feature_type
    results['3'] = 'Accuracy:'+str(np.round(Accuracy, 4))
    # results['4'] = 'Precision '+class_name+':'+str(np.round(Precision_P, 4))
    # results['5'] = 'Recall '+class_name+':'+str(np.round(Recall_P, 4))
    # results['6'] = 'F-measure '+class_name+':'+str(np.round(F1_P, 4))
    # results['7'] = 'Precision non-'+class_name+':'+str(np.round(Precision_N, 4))
    # results['8'] = 'Recall non-'+class_name+':'+str(np.round(Recall_N, 4))
    # results['9'] = 'F-measure non-'+class_name+':'+str(np.round(F1_N, 4))
    results['4'] = 'Precision sp'+':'+str(np.round(Precision_P, 4))
    results['5'] = 'Recall sp'+':'+str(np.round(Recall_P, 4))
    results['6'] = 'F-measure sp'+':'+str(np.round(F1_P, 4))
    results['7'] = 'Precision mu'+':'+str(np.round(Precision_N, 4))
    results['8'] = 'Recall mu'+':'+str(np.round(Recall_N, 4))
    results['9'] = 'F-measure mu'+':'+str(np.round(F1_N, 4))
    misc.print_analysis(opFile, results)
    
    print('Segment level statistics %s: A=%.2f, P=%.4f, R=%.4f, F=%.4f\n' % (class_name, Accuracy, Precision_P, Recall_P, F1_P))
    print('Segment level statistics non-%s: A=%.2f, P=%.4f, R=%.4f, F=%.4f\n' % (class_name, Accuracy, Precision_N, Recall_N, F1_N))



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

    # ref_event_frms_count = np.zeros(len(ref_event_frms))
    # frm_diff_min = []
    # if np.size(est_event_frms)>0:
    #     for i in range(len(est_event_frms)):
    #         TP_flag = 0
    #         frm_diff = []
    #         for j in range(len(ref_event_frms)):
    #             frm_diff.append(np.abs(est_event_frms[i]-ref_event_frms[j]))
    #             if  np.abs(est_event_frms[i]-ref_event_frms[j])<=twin_frms:
    #                 TP_flag = j+1
    #                 ref_event_frms_count[j] += 1
    #                 break
    #         frm_diff_min.append(np.min(frm_diff))
    #         if (TP_flag>0):
    #             if ref_event_frms_count[TP_flag-1]==1:
    #                 TP += 1
    #             else:
    #                 FP += 1
    #         else:
    #             FP += 1
    # FN = len(ref_event_frms)-TP # np.sum(ref_event_frms_count==0)

    if np.size(ref_event_frms)>0:
        for j in range(len(ref_event_frms)):
            FN_flag = True
            for i in range(len(est_event_frms)):
                if  np.abs(est_event_frms[i]-ref_event_frms[j])<=twin_frms:
                    FN_flag = False
                    break
            if FN_flag:
                FN += 1
            else:
                TP += 1
    FP = len(est_event_frms)-TP

    # print('Average frame differences: ', np.mean(frm_diff_min), np.std(frm_diff_min))

    return TP, FP, FN



def get_final_statistics(TPc, FPc, FNc, Nc, opFile, class_name, fold, feature_type, twin, segFunc):
    # print('get_final_statistics: ', class_name, twin, TPc, FPc, FNc, Nc)
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
    results['4'] = 'SegFunc:'+segFunc
    results['5'] = 'True Positives:'+str(TPc)
    results['6'] = 'False Positives:'+str(FPc)
    results['7'] = 'False Negatives:'+str(FNc)
    results['8'] = 'Precision:'+str(np.round(Pc, 4))
    results['9'] = 'Recall:'+str(np.round(Rc, 4))
    results['10'] = 'F-measure:'+str(np.round(Fc, 4))
    results['11'] = 'Deletion Rate:'+str(np.round(Dc, 4))
    results['12'] = 'Insertion Rate:'+str(np.round(Ic, 4))
    results['13'] = 'Error Rate:'+str(np.round(Ec, 4))
    misc.print_analysis(opFile, results)
    
    return Pc, Rc, Fc, Dc, Ic, Ec




def compute_segmentation_performance(event_type, Statistics, PtdLabels_sp, PtdLabels_mu, ref_speech_marker, ref_music_marker, tol_windows, win_size=5):
    ref_speech_marker = np.squeeze(ref_speech_marker)
    ref_music_marker = np.squeeze(ref_music_marker)
    PtdLabels_sp = np.squeeze(PtdLabels_sp)
    PtdLabels_mu = np.squeeze(PtdLabels_mu)
    for twin in tol_windows:
        nFrames = len(ref_speech_marker)
        # print('nFrames = ', nFrames, twin)
        # print('ref_speech_marker: ', np.shape(ref_speech_marker), np.unique(ref_speech_marker))
        # print('ref_music_marker: ', np.shape(ref_music_marker), np.unique(ref_music_marker))
        # print('PtdLabels_sp: ', np.shape(PtdLabels_sp), np.unique(PtdLabels_sp))
        # print('PtdLabels_mu: ', np.shape(PtdLabels_mu), np.unique(PtdLabels_mu))
        
        # d = np.shape(ref_speech_marker)[0]-np.shape(PtdLabels_sp)[0]
        # ref_speech_marker = ref_speech_marker[d:np.shape(PtdLabels_sp)[0]]
        # print('ref_speech_marker modified: ', d, np.shape(ref_speech_marker), np.unique(ref_speech_marker))
        # d = np.shape(ref_music_marker)[0]-np.shape(PtdLabels_mu)[0]
        # ref_music_marker = ref_music_marker[d:np.shape(PtdLabels_mu)[0]]
        # print('ref_music_marker modified: ', d, np.shape(ref_music_marker), np.unique(ref_music_marker))

        other_idx_sp = np.squeeze(np.where(ref_speech_marker<0))
        ref_speech_marker[other_idx_sp] = 0
        # print('ref_speech_marker modified: ', np.shape(ref_speech_marker), np.unique(ref_speech_marker))
        PtdLabels_sp[other_idx_sp] = 0
        # print('PtdLabels_sp modified: ', np.shape(PtdLabels_sp), np.unique(PtdLabels_sp))
        
        other_idx_mu = np.squeeze(np.where(ref_music_marker<0))
        ref_music_marker[other_idx_mu] = 0
        # print('ref_music_marker modified: ', np.shape(ref_music_marker), np.unique(ref_music_marker))
        PtdLabels_mu[other_idx_mu] = 0
        # print('PtdLabels_mu modified: ', np.shape(PtdLabels_mu), np.unique(PtdLabels_mu))

        if event_type=='onoff':
            ref_speech_marker_diff = np.abs(np.diff(ref_speech_marker))
            ref_music_marker_diff = np.abs(np.diff(ref_music_marker))
            est_speech_marker_diff = np.abs(np.diff(PtdLabels_sp))
            est_music_marker_diff = np.abs(np.diff(PtdLabels_mu))
        elif event_type=='on':
            ref_speech_marker_diff = (np.diff(ref_speech_marker)>0).astype(int)
            ref_music_marker_diff = (np.diff(ref_music_marker)>0).astype(int)
            est_speech_marker_diff = (np.diff(PtdLabels_sp)>0).astype(int)
            est_music_marker_diff = (np.diff(PtdLabels_mu)>0).astype(int)
        
        # print('ref_speech_marker_diff: ', np.shape(ref_speech_marker_diff), np.unique(ref_speech_marker_diff))
        # print('ref_music_marker_diff: ', np.shape(ref_music_marker_diff), np.unique(ref_music_marker_diff))
        # print('est_speech_marker_diff: ', np.shape(est_speech_marker_diff), np.unique(est_speech_marker_diff))
        # print('est_music_marker_diff: ', np.shape(est_music_marker_diff), np.unique(est_music_marker_diff))

        twin_frms = int(np.ceil((twin/(nFrames*10+15))*nFrames))
        TP_sp, FP_sp, FN_sp = get_basic_statistics(twin_frms, ref_speech_marker_diff, est_speech_marker_diff)
        Statistics[twin]['TP_sp'] += TP_sp
        Statistics[twin]['FP_sp'] += FP_sp
        Statistics[twin]['FN_sp'] += FN_sp
        Statistics[twin]['N_sp'] += np.sum(ref_speech_marker_diff)
        # print('%d Basic stats sp(%d): N_ref=%d, N_est=%d, TP=%d, FP=%d, FN=%d' % (twin, len(ref_speech_marker_diff), np.sum(ref_speech_marker_diff), np.sum(est_speech_marker_diff), TP_sp, FP_sp, FN_sp), end='\t', flush=True)

        TP_mu, FP_mu, FN_mu = get_basic_statistics(twin_frms, ref_music_marker_diff, est_music_marker_diff)
        Statistics[twin]['TP_mu'] += TP_mu
        Statistics[twin]['FP_mu'] += FP_mu
        Statistics[twin]['FN_mu'] += FN_mu
        Statistics[twin]['N_mu'] += np.sum(ref_music_marker_diff)
        # print('%d Basic stats mu (%d): N_ref=%d, N_est=%d, TP=%d, FP=%d, FN=%d' % (twin, len(ref_music_marker_diff), np.sum(ref_music_marker_diff), np.sum(est_music_marker_diff), TP_mu, FP_mu, FN_mu))
        
        prec_sp = TP_sp/(TP_sp+FP_sp+1e-10)
        rec_sp = TP_sp/(TP_sp+FN_sp+1e-10)
        fscore_sp = 2*prec_sp*rec_sp/(prec_sp+rec_sp+1e-10)
        prec_mu = TP_mu/(TP_mu+FP_mu+1e-10)
        rec_mu = TP_mu/(TP_mu+FN_mu+1e-10)
        fscore_mu = 2*prec_mu*rec_mu/(prec_mu+rec_mu+1e-10)
        # print(twin, fl, '\tFscore: ', prec_sp, rec_sp, fscore_sp, prec_mu, rec_mu, fscore_mu)
        # print('Win=(%d,%d,%d)\t(P_sp:%.4f, R_sp:%.4f, F1_sp:%.4f)\t(P_mu:%.4f, R_mu:%.4f, F1_mu:%.4f): ' % (twin, twin_frms, nFrames, prec_sp, rec_sp, fscore_sp, prec_mu, rec_mu, fscore_mu))
    # print('Stats: ', Stats)
    
    return Statistics.copy()



def print_segmentation_results(tolerance_windows, smoothing_win_size, opDir, event_type, feature_type, fold, plot_fig=False, segmentation_track=False, **kwargs):
    if 'segFunc' in kwargs.keys():
        segFunc = kwargs['segFunc']
    else:
        segFunc = 'LSTM'
    
    segmentation_path = opDir + '/__segmentation_results/' + feature_type + '/fold' + str(fold) + '/'
    seg_files = librosa.util.find_files(segmentation_path, ext=['mat'])
    classification_path = opDir + '/__Frame_Predictions_CNN/' + feature_type + '/'
    prediction_files = librosa.util.find_files(classification_path, ext=['pkl'])
    # print('segmentation_path: ', segmentation_path)
    # print('seg_files: ', seg_files)
    
    Statistics = None
    Statistics = {twin:{'TP_sp':0, 'FP_sp':0, 'FN_sp':0, 'N_sp':0, 'TP_mu':0, 'FP_mu':0, 'FN_mu':0, 'N_mu':0} for twin in tolerance_windows}
    for fl in seg_files:        
        matching_files = [fl.split('/')[-1].split('.')[0]+'_fold'+str(fold) in pfl.split('/')[-1].split('.')[0] for pfl in prediction_files]
        matching_idx = np.squeeze(np.where(matching_files))
        # print('matching_files: ', matching_files, matching_idx)
        pred_fl = prediction_files[matching_idx]
        # print('\n\n\npred_fl: ', pred_fl.split('/')[-1], fl.split('/')[-1])
        CNN_Frame_Predictions = misc.load_obj(classification_path, pred_fl.split('/')[-1].split('.')[0])
        
        # print('Filename: ', fl)
        Segmentation_Results = loadmat(fl)
        # print(Segmentation_Results.keys())
        MU_MARKER = Segmentation_Results['music_groundtruth']
        SP_MARKER = Segmentation_Results['speech_groundtruth']

        # MU_MARKER = mode_filtering(MU_MARKER, smoothing_win_size)
        # SP_MARKER = mode_filtering(SP_MARKER, smoothing_win_size)

        cnn_pred = CNN_Frame_Predictions['Pred']
        if smoothing_win_size>0:
            cnn_pred[:,0] = medfilt(cnn_pred[:,0], smoothing_win_size)
            cnn_pred[:,1] = 1-cnn_pred[:,0]
        CNN_PtdLabels_mu = np.argmin(cnn_pred, axis=1)
        CNN_PtdLabels_sp = np.argmax(cnn_pred, axis=1)
        # print('CNN_Frame_Predictions: ', np.shape(MU_MARKER), np.shape(SP_MARKER), np.shape(CNN_PtdLabels_mu), np.shape(CNN_PtdLabels_sp))
        print('CNN accuracy (mu): ', np.sum(np.array(MU_MARKER[:,0])==np.array(CNN_PtdLabels_mu))/len(MU_MARKER))
        print('CNN accuracy (sp): ', np.sum(np.array(SP_MARKER[:,0])==np.array(CNN_PtdLabels_sp))/len(SP_MARKER))

        if segFunc=='LSTM':
            lstm_pred = Segmentation_Results['LSTM_Predictions']
            if smoothing_win_size>0:
                lstm_pred[:,0] = medfilt(lstm_pred[:,0], smoothing_win_size)
                lstm_pred[:,1] = 1-lstm_pred[:,0]            
            LSTM_PtdLabels_mu = np.argmin(lstm_pred, axis=1)
            LSTM_PtdLabels_sp = np.argmax(lstm_pred, axis=1)
            # print('LSTM_Segmentation_Results: ', np.shape(LSTM_PtdLabels_mu), np.shape(LSTM_PtdLabels_sp))
            # print('Labels: sp', np.unique(SP_MARKER), ' mu', np.unique(MU_MARKER))
            Statistics = compute_segmentation_performance(event_type, Statistics.copy(), LSTM_PtdLabels_sp, LSTM_PtdLabels_mu, SP_MARKER, MU_MARKER, tolerance_windows, win_size=smoothing_win_size)
            if plot_fig:
                draw_figures(opDir, feature_type, fold, segmentation_track, fl, MU_MARKER, CNN_PtdLabels_mu, LSTM_PtdLabels_mu, CNN_Frame_Predictions, SP_MARKER, CNN_PtdLabels_sp, LSTM_PtdLabels_sp, smoothing_win_size)
            
        elif segFunc=='HMM':
            HMM_PtdLabels_sp = Segmentation_Results['HMM_PtdLabels']
            if smoothing_win_size>0:
                HMM_PtdLabels_sp = mode_filtering(HMM_PtdLabels_sp, smoothing_win_size)
            
            # HMM_PtdLabels_sp = np.squeeze(np.roll(np.array(HMM_PtdLabels_sp, ndmin=2).T, -2*smoothing_win_size, axis=0))
            
            HMM_PtdLabels_mu = 1-HMM_PtdLabels_sp
            # print('HMM_Segmentation_Results: ', np.shape(HMM_PtdLabels_mu), np.shape(HMM_PtdLabels_sp))
            # print('Labels: sp', np.unique(SP_MARKER), ' mu', np.unique(MU_MARKER))

            print('HMM accuracy (mu): ', np.sum(np.array(MU_MARKER[:,0])==np.array(HMM_PtdLabels_mu))/len(MU_MARKER))
            print('HMM accuracy (sp): ', np.sum(np.array(SP_MARKER[:,0])==np.array(HMM_PtdLabels_sp))/len(SP_MARKER))


            Statistics = compute_segmentation_performance(event_type, Statistics.copy(), HMM_PtdLabels_sp, HMM_PtdLabels_mu, SP_MARKER, MU_MARKER, tolerance_windows, win_size=smoothing_win_size)
            if plot_fig:
                draw_figures(opDir, feature_type, fold, segmentation_track, fl, MU_MARKER, CNN_PtdLabels_mu, HMM_PtdLabels_mu, CNN_Frame_Predictions, SP_MARKER, CNN_PtdLabels_sp, HMM_PtdLabels_sp, smoothing_win_size)


        elif segFunc=='CNN':
            # print('CNN_Segmentation_Results: ', np.shape(CNN_PtdLabels_mu), np.shape(CNN_PtdLabels_sp))
            # print('Labels: sp', np.unique(SP_MARKER), ' mu', np.unique(MU_MARKER))

            CNN_PtdLabels_sp = np.squeeze(np.roll(np.array(CNN_PtdLabels_sp, ndmin=2).T, -2000, axis=0))
            CNN_PtdLabels_mu = 1-CNN_PtdLabels_sp

            Statistics = compute_segmentation_performance(event_type, Statistics.copy(), CNN_PtdLabels_sp, CNN_PtdLabels_mu, SP_MARKER, MU_MARKER, tolerance_windows, win_size=smoothing_win_size)
            if plot_fig:
                draw_figures(opDir, feature_type, fold, segmentation_track, fl, MU_MARKER, CNN_PtdLabels_mu, CNN_PtdLabels_mu, CNN_Frame_Predictions, SP_MARKER, CNN_PtdLabels_sp, CNN_PtdLabels_sp, smoothing_win_size)
            
        
        # print(Statistics)


    opFile = opDir + '/Event_level_results.csv'
    for twin in tolerance_windows:
        P_sp, R_sp, F_sp, D_sp, I_sp, E_sp = get_final_statistics(Statistics[twin]['TP_sp'], Statistics[twin]['FP_sp'], Statistics[twin]['FN_sp'], Statistics[twin]['N_sp'], opFile, 'speech', fold, feature_type, twin, segFunc)
        
    for twin in tolerance_windows:
        P_mu, R_mu, F_mu, D_mu, I_mu, E_mu = get_final_statistics(Statistics[twin]['TP_mu'], Statistics[twin]['FP_mu'], Statistics[twin]['FN_mu'], Statistics[twin]['N_mu'], opFile, 'music', fold, feature_type, twin, segFunc)
        
    for twin in tolerance_windows:
        P, R, F, D, I, E = get_final_statistics(
            Statistics[twin]['TP_sp']+Statistics[twin]['TP_mu'], 
            Statistics[twin]['FP_sp']+Statistics[twin]['FP_mu'], 
            Statistics[twin]['FN_sp']+Statistics[twin]['FN_mu'], 
            Statistics[twin]['N_sp']+Statistics[twin]['N_mu'], 
            opFile, 'overall', fold, feature_type, twin, segFunc)
        
    # SP_MARKER = []
    # PtdLabels_sp = []
    # for fl in seg_files:
    #     Segmentation_Results = None
    #     Segmentation_Results = loadmat(fl)
    #     ref_marker = Segmentation_Results['speech_groundtruth']
    #     # ref_marker = mode_filtering(ref_marker, smoothing_win_size)

    #     if segFunc=='LSTM':
    #         lstm_pred = Segmentation_Results['LSTM_Predictions']
    #         if smoothing_win_size>0:
    #             lstm_pred[:,0] = medfilt(lstm_pred[:,0], smoothing_win_size)
    #             lstm_pred[:,1] = 1-lstm_pred[:,0]            
    #         est_marker = np.argmax(lstm_pred, axis=1)
    #     elif segFunc=='HMM':
    #         est_marker = Segmentation_Results['HMM_PtdLabels']
    #         if smoothing_win_size>0:
    #             est_marker = mode_filtering(est_marker, smoothing_win_size)
    #     elif segFunc=='CNN':
    #         cnn_pred = Segmentation_Results['CNN_Predictions']
    #         if smoothing_win_size>0:
    #             cnn_pred[:,0] = medfilt(cnn_pred[:,0], smoothing_win_size)
    #             cnn_pred[:,1] = 1-cnn_pred[:,0]        
    #         est_marker = np.argmax(cnn_pred, axis=1)
        
    #     ref_marker = np.squeeze(ref_marker)
    #     est_marker = np.squeeze(est_marker)
    #     # print('ref_marker: ', np.shape(ref_marker), np.unique(ref_marker))
    #     # print('est_marker: ', np.shape(est_marker), np.unique(est_marker))
    #     non_other_idx = np.squeeze(np.where(ref_marker>=0))
    #     # print(np.shape(ref_marker), np.shape(est_marker), np.shape(non_other_idx))
    #     if np.size(non_other_idx)<=1:
    #         continue
    #     ref_marker = ref_marker[non_other_idx]
    #     est_marker = est_marker[non_other_idx]
    #     # print('ref_marker modified: ', np.shape(ref_marker), np.unique(ref_marker))
    #     # print('est_marker modified: ', np.shape(est_marker), np.unique(est_marker))
        
    #     acc = np.round(np.sum(np.array(est_marker)==np.array(ref_marker))/np.size(ref_marker),4)
    #     print('sp: ', fl.split('/')[-1], acc)
        
    #     SP_MARKER.extend(ref_marker)
    #     PtdLabels_sp.extend(est_marker)
    # opFile = opDir + '/Segment_level_results.csv'
    # get_segment_level_statistics(SP_MARKER, PtdLabels_sp, 'speech', fold, opFile, feature_type)
        

    # MU_MARKER = []
    # PtdLabels_mu = []
    # for fl in seg_files:
    #     Segmentation_Results = None
    #     Segmentation_Results = loadmat(fl)
    #     ref_marker = Segmentation_Results['music_groundtruth']
    #     # ref_marker = mode_filtering(ref_marker, smoothing_win_size)

    #     if segFunc=='LSTM':
    #         lstm_pred = Segmentation_Results['LSTM_Predictions']
    #         if smoothing_win_size>0:
    #             lstm_pred[:,0] = medfilt(lstm_pred[:,0], smoothing_win_size)
    #             lstm_pred[:,1] = 1-lstm_pred[:,0]            
    #         est_marker = np.argmin(lstm_pred, axis=1)
    #     elif segFunc=='HMM':
    #         est_marker = 1-Segmentation_Results['HMM_PtdLabels']
    #         if smoothing_win_size>0:
    #             est_marker = mode_filtering(est_marker, smoothing_win_size)
    #     elif segFunc=='CNN':
    #         cnn_pred = Segmentation_Results['CNN_Predictions']
    #         if smoothing_win_size>0:
    #             cnn_pred[:,0] = medfilt(cnn_pred[:,0], smoothing_win_size)
    #             cnn_pred[:,1] = 1-cnn_pred[:,0]        
    #         est_marker = np.argmin(cnn_pred, axis=1)

    #     ref_marker = np.squeeze(ref_marker)
    #     est_marker = np.squeeze(est_marker)
    #     # print('ref_marker: ', np.shape(ref_marker), np.unique(ref_marker))
    #     # print('est_marker: ', np.shape(est_marker), np.unique(est_marker))
    #     non_other_idx = np.squeeze(np.where(ref_marker>=0))
    #     if np.size(non_other_idx)<=1:
    #         continue
    #     ref_marker = ref_marker[non_other_idx]
    #     est_marker = est_marker[non_other_idx]
    #     # print('ref_marker modified: ', np.shape(ref_marker), np.unique(ref_marker))
    #     # print('est_marker modified: ', np.shape(est_marker), np.unique(est_marker))

    #     acc = np.round(np.sum(np.array(est_marker)==np.array(ref_marker))/np.size(ref_marker),4)
    #     print('mu: ', fl.split('/')[-1], acc)

    #     MU_MARKER.extend(ref_marker)
    #     PtdLabels_mu.extend(est_marker)
    # opFile = opDir + '/Segment_level_results.csv'
    # get_segment_level_statistics(MU_MARKER, PtdLabels_mu, 'music', fold, opFile, feature_type)

    
