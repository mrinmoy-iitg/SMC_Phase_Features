#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:02:04 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import datetime
import numpy as np
from scipy.io import savemat
import time
import csv
import librosa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.metrics import  confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
import lib.classifier.cnn_classifier as CNN
import lib.feature.preprocessing as preproc
from lib.segmentation.cython_funcs import extract_patches as cextract_patches
import lib.misc as misc
from lib.segmentation.segmentation_performance import print_segmentation_results
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture




def start_GPU_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 2, 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    K.clear_session()
    tf.compat.v1.keras.backend.set_session(sess)



def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()




def plot_data_histogram(data, label, fig_opDir):
    sp_idx = np.squeeze(np.where(label==1))
    data_sp = data[sp_idx, :]
    mu_idx = np.squeeze(np.where(label==0))
    data_mu = data[mu_idx, :]
    
    sp_hist, sp_bin_edges = np.histogram(data_sp, bins=100, density=True)
    mu_hist, mu_bin_edges = np.histogram(data_mu, bins=100, density=True)
    sp_hist /= np.sum(sp_hist)
    mu_hist /= np.sum(mu_hist)
    
    plt.plot(sp_hist)
    plt.plot(mu_hist)
    plt.legend(['speech', 'music'])
    # plt.show()
    plt.savefig(fig_opDir + feature_type + '_fold' + str(fold) + '.jpg', bbox_inches='tight')
    plt.cla()





def learn_hmm(PARAMS, nMix, X, y):
    modelName = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    modelName = '.'.join(modelName.split('@'))
    model_path = '/'.join(modelName.split('/')[:-1])
    model_fName = modelName.split('/')[-1].split('.')[0]

    if not os.path.exists(model_path+'/'+model_fName+'.pkl'):
        start = time.process_time()
        mu_idx = np.squeeze(np.where(np.array(y)==0))
        X_mu = X[mu_idx,:]
        print('mu samples: ', np.size(mu_idx), np.shape(X_mu))
        sp_idx = np.squeeze(np.where(np.array(y)==1))
        X_sp = X[sp_idx,:]
        print('sp samples: ', np.size(sp_idx), np.shape(X_sp))
        
        gmm_mu = GaussianMixture(n_components=nMix, covariance_type='full')
        gmm_mu.fit(X_mu)
        print('GMM mu fitted')
        gmm_sp = GaussianMixture(n_components=nMix, covariance_type='full')
        gmm_sp.fit(X_sp)
        print('GMM sp fitted')
    
        means_ = np.append(np.expand_dims(gmm_mu.means_, axis=0), np.expand_dims(gmm_sp.means_, axis=0), axis=0) 
        weights_ = np.append(np.array(gmm_mu.weights_, ndmin=2), np.array(gmm_sp.weights_, ndmin=2), axis=0)
        covars_ = np.append(np.expand_dims(gmm_mu.covariances_, axis=0), np.expand_dims(gmm_sp.covariances_, axis=0), axis=0)
        print('means_: ', np.shape(means_))
        print('weights_: ', np.shape(weights_))
        print('covars_: ', np.shape(covars_))
    
        hmm_model = hmm.GMMHMM(
            n_components=2,
            n_mix=nMix,
            init_params='st', # stmcw
            params='st', # stmcw
            covariance_type='full',
            verbose=True,
            n_iter=50, 
            )
        hmm_model.weights_ = weights_
        hmm_model.means_ = means_
        hmm_model.covars_ = covars_
        hmm_model.fit(X=X)
        
        trainingTimeTaken = time.process_time() - start
        misc.save_obj({'hmm_model':hmm_model, 'trainingTimeTaken':trainingTimeTaken}, model_path, model_fName)
    else:
        hmm_model = misc.load_obj(model_path, model_fName)['hmm_model']
        trainingTimeTaken = misc.load_obj(model_path, model_fName)['trainingTimeTaken']
    print('Time taken for model training: ',trainingTimeTaken)
    
    Train_Params = {
        'model': hmm_model,
        'trainingTimeTaken': trainingTimeTaken,
        }

    return Train_Params




def test_hmm(PARAMS, Train_Params, test_data, test_label):
    start = time.process_time()
    X = test_data
    PtdLabels_test = Train_Params['model'].predict(X) #, algorithm='viterbi')
    # print('PtdLabels_test: ', np.shape(PtdLabels_test), np.shape(X), np.unique(PtdLabels_test))
    ConfMat, fscore = getPerformance(PtdLabels_test, test_label, labels=[0,1])
    print('ConfMat: ', ConfMat)
    print('Fscore: ', fscore)
    Accuracy = np.round(np.sum(np.diag(ConfMat))*100/np.sum(ConfMat),2)

    precision = ConfMat[0,0]/(ConfMat[0,0]+ConfMat[1,0]+1e-10)
    recall = ConfMat[0,0]/(ConfMat[0,0]+ConfMat[0,1]+1e-10)
    testingTimeTaken = time.process_time() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    Test_Params = {
        'PtdLabels_test': PtdLabels_test,
        'ConfMat': ConfMat,
        'fscore': fscore,
        'Accuracy': Accuracy,
        'precision': precision,
        'recall': recall,
        'testingTimeTaken': testingTimeTaken,
        }
    
    return Test_Params





def create_ngram(PARAMS, data, score_column=1):
    # print('data: ', np.shape(data))
    data_ngram = np.array(data[:,0], ndmin=2).T
    for i in range(int(PARAMS['num_steps']/2)):
        X_temp = np.roll(np.array(data[:,score_column], ndmin=2).T, -(i+1), axis=0)
        data_ngram = np.append(data_ngram, X_temp, axis=1)
    for i in range(int(PARAMS['num_steps']/2)):
        X_temp = np.roll(np.array(data[:,score_column], ndmin=2).T, (i+1), axis=0)
        data_ngram = np.append(X_temp, data_ngram, axis=1)
    # print('train data: ', np.shape(data_ngram))
    return data_ngram




def get_segmentation_data(Sequence_Data, ground_truths_sp, ground_truths_mu, data_type, files, **kwargs):
    if 'win_size' in kwargs.keys():
        win_size = kwargs['win_size']
    else:
        win_size = 0
    
    data = np.empty([])
    label = np.empty([])
    for fl in files:
        fl_data = Sequence_Data[data_type][fl]
        
        gt = np.zeros(len(ground_truths_sp[data_type][fl][:, 0]))
        sp_idx = np.squeeze(np.where(ground_truths_sp[data_type][fl][:, 0]>0))
        mu_idx = np.squeeze(np.where(ground_truths_mu[data_type][fl][:, 0]>0))
        sp_uniq, sp_counts = np.unique(ground_truths_sp[data_type][fl][:, 0], return_counts=True)
        mu_uniq, mu_counts = np.unique(ground_truths_mu[data_type][fl][:, 0], return_counts=True)
        # print('sp: ', sp_uniq, sp_counts, len(ground_truths_sp[data_type][fl][:, 0]))
        # print('mu: ', mu_uniq, mu_counts, len(ground_truths_mu[data_type][fl][:, 0]))

        gt[sp_idx] = 1
        gt[mu_idx] = 0

        if (win_size>0):
            fl_data[:,0] = medfilt(fl_data[:,0], win_size)
            fl_data[:,1] = 1 - fl_data[:,0]
            
        if (data_type=='train'):
            gt = np.argmax(fl_data, axis=1)
                
        # fl_data = create_ngram(PARAMS, fl_data, score_column=1)
        fl_data = fl_data[:,1].reshape(-1,1)

        # print(fl, ' get_segmentation_data: ', np.shape(fl_data), np.shape(gt), np.unique(gt))

        if np.size(data)<=1:
            data = fl_data
            label = gt
        else:
            data = np.append(data, fl_data, axis=0)
            label = np.append(label, gt)
    # data = StandardScaler().fit_transform(data)
    # print('get segmentation data: ', np.shape(data), np.shape(label))

    if not np.size(data)<=1:
        pred_lab = np.argmax(data, axis=1)
        # print(np.shape(pred_lab), np.shape((pred_lab==np.array(label))))
        acc = np.sum(pred_lab==np.array(label))/len(label)
        print('Segmentation data accuracy: ', acc, np.unique(label), np.shape(data), np.shape(label))
            
    return data, label




def getPerformance(PtdLabels, GroundTruths, **kwargs):
    if 'labels' in kwargs.keys():
        labels = kwargs['labels']
    else:
        labels = np.unique(GroundTruths)
    ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels, labels=labels)
    fscore = f1_score(GroundTruths, PtdLabels, average=None, labels=labels).tolist()
    mean_fscore = np.round(np.mean(fscore), 4)
    # print('mean_fscore: ', mean_fscore)
    fscore.append(mean_fscore)

    return ConfMat, fscore




def mode_filtering(X, win_size):
    if win_size%2==0:
        win_size += 1
    X_smooth = X.copy()
    for i in range(int(win_size/2), len(X)-int(win_size/2)):
        win = X[i-int(win_size/2):i+int(win_size/2)]
        uniq_lab, uniq_counts = np.unique(win, return_counts=True)
        X_smooth[i] = uniq_lab[np.argmax(uniq_counts)]    
    return X_smooth




def plot_segmentation_results_hmm(PARAMS, opDirFig, CNN_Predictions, HMM_PtdLabels, annot_path, fl, speech_marker, music_marker, win_size):
    Frame_Labels_sp = np.argmax(CNN_Predictions, axis=1)
    Frame_Labels_mu = np.argmin(CNN_Predictions, axis=1)

    CNN_Predictions_smooth = CNN_Predictions.copy()
    if win_size>0:
        CNN_Predictions_smooth[:,0] = medfilt(CNN_Predictions_smooth[:,0], win_size)
        CNN_Predictions_smooth[:,1] = 1-CNN_Predictions_smooth[:,0]
    Frame_Labels_sp_smooth = np.argmax(CNN_Predictions_smooth, axis=1)
    Frame_Labels_mu_smooth = np.argmin(CNN_Predictions_smooth, axis=1)
    
    HMM_PtdLabels_smooth = HMM_PtdLabels.copy()
    if win_size>0:
        HMM_PtdLabels_smooth = mode_filtering(HMM_PtdLabels, win_size)

    plot_num = 0
    nPlotRows = 5
    nPlotCols = 1
    plt.figure()
    
    plot_num = 1
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    plt.plot(speech_marker, 'm-')
    plt.plot(music_marker*2, 'b-')
    plt.title('Ground Truths')
    plt.legend(['speech', 'music'])

    plot_num = 2
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    plt.plot(Frame_Labels_sp, 'm-')
    plt.plot(Frame_Labels_mu*2, 'b-')
    plt.title('Classifier labels')
    plt.legend(['speech', 'music'])

    plot_num = 3
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    plt.plot(Frame_Labels_sp_smooth, 'm-')
    plt.plot(Frame_Labels_mu_smooth*2, 'b-')
    plt.title('Smooth Classifier labels')
    plt.legend(['speech', 'music'])

    plot_num = 4
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    sp_idx = np.squeeze(np.where(HMM_PtdLabels==1))
    marker = np.zeros(len(HMM_PtdLabels))
    marker[sp_idx] = 1
    plt.plot(marker, 'm-')
    mu_idx = np.squeeze(np.where(HMM_PtdLabels==0))
    marker = np.zeros(len(HMM_PtdLabels))
    marker[mu_idx] = 2
    plt.plot(marker, 'b-')
    plt.title('HMM labels')
    plt.legend(['speech', 'music'])

    plot_num = 5
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    sp_idx = np.squeeze(np.where(HMM_PtdLabels_smooth==1))
    marker = np.zeros(len(HMM_PtdLabels_smooth))
    marker[sp_idx] = 1
    plt.plot(marker, 'm-')
    mu_idx = np.squeeze(np.where(HMM_PtdLabels_smooth==0))
    marker = np.zeros(len(HMM_PtdLabels_smooth))
    marker[mu_idx] = 2
    plt.plot(marker, 'b-')
    plt.title('HMM smooth labels')
    plt.legend(['speech', 'music'])
    
    print('figure plotting')

    # plt.show()
    plt.savefig(opDirFig+fl.split('.')[0]+'.jpg', bbox_inches='tight')




def get_annotations(folder, fl, nFrames, opDir, with_silence=True, Tw=25, Ts=10): # Designed for muspeak dataset annotations
    annot_opDir = opDir + '/__annotations/'
    if not os.path.exists(annot_opDir):
        os.makedirs(annot_opDir)
    opFile = annot_opDir+'/'+fl.split('.')[0]+'.npz'
    diff_sample_silMark = []
    if not os.path.exists(opFile):
        print('Reading annotations of ', fl)
        annotations = {}
        with open(folder+'/'+fl.split('.')[0]+'.csv', newline='\n') as csvfile:
            annotreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_count = 0
            for row in annotreader:
                if row==[]:
                    continue
                annotations[row_count] = row
                row_count += 1
        
        print('annotations: ', len(annotations))
        print(annotations)
        
        Xin, fs = librosa.core.load(folder+'/'+fl.split('.')[0]+'.wav', sr=16000)
        print('Xin: ', np.round(len(Xin)/fs,4))
        lenXin = len(Xin)/fs

        Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin, fs=fs, Tw=Tw, Ts=Ts, alpha=0.025, beta=0)

        print('Xin_silrem: ', np.round(len(Xin_silrem)/fs,4), len(sample_silMarker), np.sum(np.array(sample_silMarker))/fs)
        # lenXin = len(sample_silMarker)/fs

        if not with_silence:        
            diff_sample_silMark = np.diff(sample_silMarker)
            smpStart_list = np.squeeze(np.where(diff_sample_silMark<0))
            smpEnd_list = np.squeeze(np.where(diff_sample_silMark>0))
            print(smpStart_list)
            print(smpEnd_list)

            if np.size(smpStart_list)==1:
                smpStart_list = [smpStart_list]
            if np.size(smpEnd_list)==1:
                smpEnd_list = [smpEnd_list]
            for i in range(len(smpEnd_list)-1,0,-1):
                smpStart = smpStart_list[i]
                smpEnd = smpEnd_list[i]
                print(i, np.round(smpStart/fs,4), np.round(smpEnd/fs,4))
                lenXin -= (smpEnd-smpStart)/fs
                for row in annotations.keys():
                    tmin = float(annotations[row][0])
                    dur = float(annotations[row][1])
                    tmax = tmin+dur
                    className = annotations[row][2]
                    print('\t', annotations[row])
                    if (tmin>(smpStart/fs)) and (tmax<(smpEnd/fs)):
                        annotations[row][0] = str(smpStart/fs)
                        annotations[row][1] = str(0)
                        # annotations[row][2] = 'sil'
                        tmin = float(annotations[row][0])
                        dur = float(annotations[row][1])
                        tmax = tmin+dur
                        print('\t\t\tdeleted ', annotations[row])
                    elif (tmin>(smpStart/fs)) and (tmin<(smpEnd/fs)) and (tmax>(smpEnd/fs)):
                        annotations[row][0] = str(smpStart/fs)
                        # annotations[row][1] = str(np.max([0, dur-(smpEnd-smpStart)/fs]))
                        annotations[row][1] = str(np.max([0, dur-((smpEnd/fs)-tmin)]))
                        tmin = float(annotations[row][0])
                        dur = float(annotations[row][1])
                        tmax = tmin+dur
                        print('\t\t\t1', annotations[row])
                    elif (tmax>(smpStart/fs)) and (tmax<(smpEnd/fs)) and (tmin<(smpStart/fs)):
                        annotations[row][1] = str((smpStart/fs)-tmin)
                        dur = float(annotations[row][1])
                        tmax = tmin+dur
                        print('\t\t\t2', annotations[row])
                    elif (tmin>(smpEnd/fs)):
                        annotations[row][0] = str(np.max([0,tmin-(smpEnd-smpStart)/fs]))
                        tmin = float(annotations[row][0])
                        tmax = tmin+dur
                        print('\t\t\t3', annotations[row])
                    elif (tmin<(smpStart/fs)) and (tmax>(smpEnd/fs)):
                        annotations[row][1] = str(np.max([0, dur-((smpEnd-smpStart)/fs)]))
                        dur = float(annotations[row][1])
                        tmax = tmin+dur
                        print('\t\t\t4', annotations[row])
                print('\n')
        print(annotations)
        print('lenXin: ', len(Xin_silrem))
        speech_marker = np.zeros(nFrames)
        music_marker = np.zeros(nFrames)
        for row in annotations.keys():
            tmin = float(annotations[row][0])
            dur = float(annotations[row][1])
            if dur==0.0:
                continue
            tmax = tmin+dur
            className = annotations[row][2]
            frmStart = np.max([0, int(np.floor((tmin/lenXin)*nFrames))])
            frmEnd = np.min([int(np.ceil((tmax/lenXin)*nFrames)), nFrames-1])
            if className=='s':
                speech_marker[frmStart:frmEnd] = 1
            elif className=='m':
                music_marker[frmStart:frmEnd] = 1
            print(row, np.round(lenXin,4), np.round(tmin,4), np.round(dur,4), np.round(tmax,4), frmStart, frmEnd, className)
        
        np.savez(opFile, annotations=annotations, speech_marker=speech_marker, music_marker=music_marker)
    else:
        annotations = np.load(opFile, allow_pickle=True)['annotations']
        speech_marker = np.load(opFile, allow_pickle=True)['speech_marker']
        music_marker = np.load(opFile, allow_pickle=True)['music_marker']

    return annotations, speech_marker, music_marker



def get_feature_patches(PARAMS, FV, labels_sp, labels_mu, patch_size, patch_shift, input_shape):
    startTime = time.process_time()
    # Removing NaN and Inf
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV[~np.isnan(FV).any(axis=1), :]
        FV = FV[~np.isinf(FV).any(axis=1), :]
    else:
        FV = FV[:, ~np.isnan(FV).any(axis=0)]
        FV = FV[:, ~np.isinf(FV).any(axis=0)]

    if not PARAMS['whole_data_scaling']:
        FV = StandardScaler(copy=False).fit_transform(FV)
    # FV should be of the shape (nFeatures, nFrames)
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV.T
                
    patches = np.empty([])
    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)

    zero_padding = np.zeros((np.shape(FV)[0], int(patch_size/2)))
    FV = np.append(zero_padding, FV, axis=1)
    FV = np.append(FV, zero_padding, axis=1)
    
    patches, patch_label_sp, patch_label_mu = cextract_patches(FV, np.shape(FV), patch_size, patch_shift, labels_sp, labels_mu, 'classification')

    print('Patches: ', np.shape(patches), np.shape(patch_label_sp), np.shape(patch_label_mu))
    
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
    
    finishTime = time.process_time()
    print('get_feature_patches Cython time taken: ', finishTime-startTime)
    return patches, patch_label_sp, patch_label_mu





def load_data_from_files(PARAMS, featName, files, mean, stdev, patch_shift):
    label_sp = []
    label_mu = []
    data = np.empty([])
    # print('load_data_from_files files: ', files, PARAMS['test_classes'])
    for clNum in PARAMS['test_classes'].keys():
        class_name = PARAMS['test_classes'][clNum]
        for fl in files[class_name]:
            fName = PARAMS['folder'] + '/' + featName + '/' + class_name + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
            print('load_data_from_files: ', fName, os.path.exists(fName))
            if not os.path.exists(fName):
                continue
            FV = np.load(fName, allow_pickle=True)
            
            ''' Whole data scaling '''
            if PARAMS['whole_data_scaling']:
                FV = np.subtract(FV, np.repeat(np.array(mean, ndmin=2), np.shape(FV)[0], axis=0))
                FV = np.divide(FV, np.repeat(np.array(stdev+1e-10, ndmin=2), np.shape(FV)[0], axis=0))
                # print('Whole data scaling FV: ', np.shape(FV))
            
            nFrames = np.shape(FV)[0]
            annotations, speech_marker, music_marker = get_annotations(PARAMS['annot_path'], fl.split('/')[-1], nFrames, PARAMS['opDir'], with_silence=False)

            FV_patches, speech_marker, music_marker = get_feature_patches(PARAMS, FV, speech_marker, music_marker, PARAMS['CNN_patch_size'], patch_shift, PARAMS['input_shape'][featName])
            FV_patches = np.expand_dims(FV_patches, axis=3)            
            # print(fl, np.shape(FV_patches), np.shape(speech_marker), np.shape(music_marker))

            if np.size(data)<=1:
                data = FV_patches
            else:
                # print('data: ', np.shape(data), np.shape(FV))
                data = np.append(data, FV_patches, 0)
            label_sp.extend(speech_marker)
            label_mu.extend(music_marker)
    label_sp = np.array(label_sp, ndmin=2).T
    label_mu = np.array(label_mu, ndmin=2).T

    return data, label_sp, label_mu




def frame_probabilities_generator_test(PARAMS, folder, fl, Ensemble_Train_Params, pred_opDir):
    startTime = time.process_time()
    Predictions_Ensemble = np.empty([])
    count = -1
    
    count += 1
    labels_sp = []
    labels_mu = []
    print('Ensemble features: ', Ensemble_Train_Params.keys())
    for featName in Ensemble_Train_Params.keys():
        print('\n\n\n', fl, featName)
        pred_opDir_feat = pred_opDir + '/' + featName + '/'
        if not os.path.exists(pred_opDir_feat):
            os.makedirs(pred_opDir_feat)
        pred_fName = fl.split('.')[0] + '_fold' + str(PARAMS['fold']) +  '_Predictions'
        lab_fName = fl.split('.')[0] + '_fold' + str(PARAMS['fold']) +  '_Labels'
        if not os.path.exists(pred_opDir_feat+pred_fName+'.pkl'):
            Train_Params = Ensemble_Train_Params[featName]
            
            data_stats_dir = PARAMS['cnn_model_path'] + '/__data_stats/'
            data_stats_fName = featName + '_fold' + str(PARAMS['fold']) + '_mean_stdev'
            # print('data stats fName: ', data_stats_dir + '/' + data_stats_fName + '.pkl')
            if os.path.exists(data_stats_dir + '/' + data_stats_fName + '.pkl'):
                data_mean = misc.load_obj(data_stats_dir, data_stats_fName)['mean']
                data_stdev = misc.load_obj(data_stats_dir, data_stats_fName)['stdev']
                # print(featName, ' data stats loaded')
                fName = folder + '/' + featName + '/' + PARAMS['test_classes'][0] + '/' + fl.split('.')[0] + '.npy'
                print('fName: ', fName)
                FV_patches, labels_sp, labels_mu = load_data_from_files(PARAMS, featName, {PARAMS['test_classes'][0]:[fName]}, data_mean, data_stdev, PARAMS['CNN_patch_shift_test'])
            else:
                print(featName, ' data stats not available')
                continue
            print(featName, 'count, FV_patches, labels_sp, labels_mu: ', count, np.shape(FV_patches), np.shape(labels_sp), np.shape(labels_mu))
            pred = Train_Params['model'].predict(x=FV_patches, verbose=1)            
            pred_lab_sp = np.argmax(pred, axis=1)
            pred_lab_mu = np.argmin(pred, axis=1)
            acc_sp = np.sum(pred_lab_sp==labels_sp)/len(labels_sp)
            acc_mu = np.sum(pred_lab_mu==labels_mu)/len(labels_mu)
            print('pred shape: ', np.shape(pred), acc_sp, acc_mu)

            misc.save_obj(pred, pred_opDir_feat, pred_fName)
            misc.save_obj(labels_sp, pred_opDir_feat, lab_fName+'_sp')
            misc.save_obj(labels_mu, pred_opDir_feat, lab_fName+'_mu')
            print('Test predictions saved!!!')
        else:
            pred = misc.load_obj(pred_opDir_feat, pred_fName)
            if os.path.exists(pred_opDir_feat+lab_fName+'_sp.pkl'):
                labels_sp = misc.load_obj(pred_opDir_feat, lab_fName+'_sp')
                labels_mu = misc.load_obj(pred_opDir_feat, lab_fName+'_mu')
                print('Test predictions loaded!!!', np.shape(pred), np.shape(labels_sp), np.shape(labels_mu))
            else:
                print('Test predictions loaded!!!', np.shape(pred))



        if not os.path.exists(pred_opDir_feat+lab_fName+'.pkl'):
            Train_Params = Ensemble_Train_Params[featName]
            
            data_stats_dir = PARAMS['cnn_model_path'] + '/__data_stats/'
            data_stats_fName = featName + '_fold' + str(PARAMS['fold']) + '_mean_stdev'
            if os.path.exists(data_stats_dir + '/' + data_stats_fName + '.pkl'):
                data_mean = misc.load_obj(data_stats_dir, data_stats_fName)['mean']
                data_stdev = misc.load_obj(data_stats_dir, data_stats_fName)['stdev']
                fName = folder + '/' + featName + '/' + PARAMS['test_classes'][0] + '/' + fl.split('.')[0] + '.npy'
                print('fName: ', fName)
                FV_patches, labels_sp, labels_mu = load_data_from_files(PARAMS, featName, {PARAMS['test_classes'][0]:[fName]}, data_mean, data_stdev, PARAMS['CNN_patch_shift_test'])
            else:
                print(featName, ' data stats not available')
                continue
            misc.save_obj(labels_sp, pred_opDir_feat, lab_fName+'_sp')
            misc.save_obj(labels_mu, pred_opDir_feat, lab_fName+'_mu')
            print('Labels saved!!!', np.shape(labels_sp), np.shape(labels_mu))
        else:
            labels_sp = misc.load_obj(pred_opDir_feat, lab_fName+'_sp')
            labels_mu = misc.load_obj(pred_opDir_feat, lab_fName+'_mu')
            print('Labels loaded!!!', np.shape(labels_sp), np.shape(labels_mu))



        pred_lab_sp = np.argmax(pred, axis=1)
        pred_lab_mu = np.argmin(pred, axis=1)
        print('Total shapes: ', np.shape(pred_lab_sp), np.shape(labels_sp), np.shape(pred_lab_mu), np.shape(labels_mu))
        acc_sp = np.round(np.sum(np.array(pred_lab_sp)==np.array(labels_sp))/np.size(pred_lab_sp),4)
        acc_mu = np.round(np.sum(np.array(pred_lab_mu)==np.array(labels_mu))/np.size(pred_lab_mu),4)
        print(featName, ' Total accuracy: ', acc_sp, acc_mu)

        if np.size(Predictions_Ensemble)<=1:
            Predictions_Ensemble = np.array(pred, ndmin=2)
        else:
            print('Predictions: ', np.shape(Predictions_Ensemble), np.shape(pred), len(Ensemble_Train_Params))
            if np.shape(pred)[0]<np.shape(Predictions_Ensemble)[0]:
                d = np.shape(Predictions_Ensemble)[0] - np.shape(pred)[0]
                pred = np.append(pred, pred[-d:,:], axis=0)
            else:
                pred = pred[:np.shape(Predictions_Ensemble)[0], :]
            Predictions_Ensemble = np.add(Predictions_Ensemble, np.array(pred, ndmin=2))
    Predictions_Ensemble /= len(Ensemble_Train_Params)
    # print('Mean predictions: ', np.shape(Predictions_Ensemble), np.mean(Predictions_Ensemble, axis=0), end='\n', flush=True)

    probability_genTime = time.process_time() - startTime
    minFrms = np.min([np.shape(Predictions_Ensemble)[0], np.shape(labels_sp)[0], np.shape(labels_mu)[0]])
    Predictions_Ensemble = Predictions_Ensemble[:minFrms, :]
    labels_sp = labels_sp[:minFrms] 
    labels_mu = labels_mu[:minFrms]

    return Predictions_Ensemble, labels_sp, labels_mu, probability_genTime




def get_test_predictions(PARAMS, Ensemble_Train_Params, feature_type, fileName):
    pred_opDir = PARAMS['cnn_model_path'] + '/__Frame_Predictions_CNN/'
    if not os.path.exists(pred_opDir):
        os.makedirs(pred_opDir)
    pred_opDir_feat = pred_opDir + '/' + feature_type + '/'
    if not os.path.exists(pred_opDir_feat):
        os.makedirs(pred_opDir_feat)
    pred_fName = fileName.split('.')[0] + '_fold' + str(PARAMS['fold']) + '_Predictions_Dict'
    
    if not os.path.exists(pred_opDir_feat + pred_fName + '.pkl'):
        Pred, Labels_sp, Labels_mu, probability_genTime = frame_probabilities_generator_test(PARAMS, PARAMS['test_path'], fileName, Ensemble_Train_Params, pred_opDir)
        print('get_test_predictions: ', np.shape(Pred), np.shape(Labels_sp), np.shape(Labels_mu))
        Predictions_Dict = {
            'Pred': Pred,
            'Labels_sp': Labels_sp,
            'Labels_mu': Labels_mu,
            'probability_genTime': probability_genTime
            }
        if PARAMS['save_flag']:
            misc.save_obj(Predictions_Dict, pred_opDir_feat, pred_fName)
    else:
        # print(fileName, ' predictions available!!!')
        Predictions_Dict = misc.load_obj(pred_opDir_feat, pred_fName)

    return Predictions_Dict




def load_model(modelName, input_shape):
    modelName = '@'.join(modelName.split('.')[:-1]) + '.' + modelName.split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
        
    learning_rate = 0.0001
    if os.path.exists(paramFile):
        epochs = np.load(paramFile)['epochs']
        batch_size = np.load(paramFile)['batch_size']
        try:
            learning_rate = np.load(paramFile)['lr']
        except:
            learning_rate = 0.0001
        trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        optimizer = optimizers.Adam(lr=learning_rate)
        
        try:
            with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
                source_model = model_from_json(f.read())
        except:
            source_model, learning_rate = CNN.get_cnn_model(input_shape, 2)
        source_model.load_weights(weightFile) # Load weights into the new model
        source_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('\n\t\t\tCNN model exists! Loaded. Training time required=',trainingTimeTaken)
        print(source_model.summary())
        
        Train_Params = {
            'model': source_model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    else:
        print(paramFile)
        print('CNN model does not exists!')
        Train_Params = {}
        
    return Train_Params




def get_data_stats(PARAMS, featName, files):
    mean_sp = np.empty([], dtype=np.float128)
    nSpFrames = 0
    mean_mu = np.empty([], dtype=np.float128)
    nMuFrames = 0
    for clNum in PARAMS['test_classes'].keys():
        class_name = PARAMS['test_classes'][clNum]
        for fl in files[class_name]:
            fName = PARAMS['folder'] + '/' + featName + '/' + class_name + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
            print('load_data_from_files: ', fl, os.path.exists(fName))
            if not os.path.exists(fName):
                continue
            FV = np.load(fName, allow_pickle=True)
            # print('FV: ', np.shape(FV))
            FV = FV[~np.isnan(FV).any(axis=1), :]
            FV = FV[~np.isinf(FV).any(axis=1), :]
            # print('FV curated: ', np.shape(FV))
            nFrames = np.shape(FV)[0]
            annotations, speech_marker, music_marker = get_annotations(PARAMS['annot_path'], fl, nFrames, PARAMS['opDir'], with_silence=False)

            sp_idx = np.squeeze(np.where(speech_marker==1))
            nSpFrames += len(sp_idx)
            if np.size(mean_sp)<=1:
                mean_sp = np.sum(FV[sp_idx,:], axis=0)
            else:
                mean_sp += np.sum(FV[sp_idx,:], axis=0)

            mu_idx = np.squeeze(np.where(music_marker==1))
            nMuFrames += len(mu_idx)
            if np.size(mean_mu)<=1:
                mean_mu = np.sum(FV[mu_idx,:], axis=0)
            else:
                mean_mu += np.sum(FV[mu_idx,:], axis=0)
            # print('mean: ', np.shape(FV), np.shape(speech_marker), len(sp_idx), np.shape(music_marker), len(mu_idx), np.shape(mean_sp), np.shape(mean_mu))
    # print('mean_sp: ', np.round(mean_sp, 4))
    # print('mean_mu: ', np.round(mean_mu, 4))
    mean_sp /= nSpFrames
    mean_mu /= nMuFrames
    overall_mean = np.add(mean_sp, mean_mu)/2
    # print(featName, ' overall mean: ', np.round(overall_mean, 4))

    stdev = np.empty([], dtype=np.float128) # np.zeros(PARAMS['input_shape'][featName][0], dtype=np.float128)
    nFrames = 0
    for clNum in PARAMS['test_classes'].keys():
        class_name = PARAMS['test_classes'][clNum]
        for fl in files[class_name]:
            fName = PARAMS['folder'] + '/' + featName + '/' + class_name + '/' + fl.split('/')[-1].split('.')[0] + '.npy'
            if not os.path.exists(fName):
                continue
            FV = np.load(fName, allow_pickle=True)
            # print('FV: ', np.shape(FV))
            FV = FV[~np.isnan(FV).any(axis=1), :]
            FV = FV[~np.isinf(FV).any(axis=1), :]
            # print('FV curated: ', np.shape(FV))
            nFrames += np.shape(FV)[0]
            mean_arr = np.repeat(np.array(overall_mean,ndmin=2),np.shape(FV)[0],axis=0)
            if np.size(stdev)<=1:
                stdev = np.sum(np.subtract(FV, mean_arr), axis=0)
            else:
                stdev += np.sum(np.subtract(FV, mean_arr), axis=0)
            # print('stdev: ', np.shape(FV), np.shape(mean_arr), np.shape(stdev))

    stdev /= (nFrames-1)
    # print(featName, ' stdev: ', np.round(stdev, 4))
        
    return overall_mean.astype(np.float32), stdev.astype(np.float32)





def __init__():
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            # 'folder': '/scratch/mbhattacharjee/features/SMC_SPECOM/muspeak-mirex2015-detection-examples/IFDur=100frms_Tw=25ms_Ts=10ms_2020-09-14/',
            # 'model_folder': '/scratch/mbhattacharjee/features/SMC_SPECOM/musan/IFDur=100frms_Tw=25ms_Ts=10ms_2020-08-09/',
            # 'test_path': '/scratch/mbhattacharjee/features/SMC_SPECOM/muspeak-mirex2015-detection-examples/IFDur=100frms_Tw=25ms_Ts=10ms_2020-09-14/',
            # 'annot_path': '/scratch/mbhattacharjee/data/muspeak-mirex2015-detection-examples/muspeak/',
            'folder': '/home/mrinmoy/Documents/PhD_Work/Features/SMC_SPECOM/muspeak-mirex2015-detection-examples/IFDur=100frms_Tw=25ms_Ts=10ms_2020-09-14/',
            'model_folder': '/home/mrinmoy/Documents/PhD_Work/Features/SMC_SPECOM/musan/IFDur=100frms_Tw=25ms_Ts=10ms_2020-08-09/',
            'test_path': '/home/mrinmoy/Documents/PhD_Work/Features/SMC_SPECOM/muspeak-mirex2015-detection-examples/IFDur=100frms_Tw=25ms_Ts=10ms_2020-09-14/',
            'annot_path': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/muspeak-mirex2015-detection-examples_wav/muspeak/',
            'test_folder': '',
            'CNN_patch_size': 68,
            'CNN_patch_shift': 8,
            'CNN_patch_shift_test': 1,
            'clFunc': 'CNN',
            'segmentation_func': 'HMM',
            'smoothing_win_size': 1001,
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': False,
            'GPU_session':None,
            'opDir':'',
            'classes': {0:'music', 1:'speech'},
            'test_classes': {0:'wav'},
            'dataset':'',
            'opDir':'',
            'modelName':'',
            'input_dim':0,
            'Tw': 25,
            'Ts': 10,
            'feat_combinations':{
                # 'Khonglah': ['Khonglah_et_al'],
                # 'Sell': ['Sell_et_al'],
                # 'MFCC': ['MFCC-39'],
                # 'Melspectrogram': ['Melspectrogram'],
                # 'HNGDMFCC': ['HNGDMFCC'],
                # 'MGDCC': ['MGDCC'],
                # 'IFCC': ['IFCC'],
                # 'all_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'],
                'all_features-IFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC'],
                # 'all_features-MGDCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'HNGDMFCC', 'IFCC', 'Melspectrogram'],
                # 'all_features-Sell_et_all': ['Khonglah_et_al', 'MFCC-39', 'HNGDMFCC', 'IFCC', 'Melspectrogram'],
                # 'magnitude_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram'],
                # 'phase_features': ['HNGDMFCC', 'MGDCC', 'IFCC'],
                # 'phase_features-MGDCC': ['HNGDMFCC', 'IFCC'],
                }, #['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'HNGDMFCC', 'MGDCC', 'IFCC', 'Melspectrogram']
            'featName': '',
            'plot_fig': True,
            'tolerance_windows':[1000, 500],
            'num_steps': 5,
            'skip_step': 1,
            '39_dim_CC_feat': True, # section.getboolean('39_dim_CC_feat'),         
            'CNN_model_type': '__CNN_MUSAN',
            'whole_data_scaling': False,
            }

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()

    if PARAMS['test_path']=='':
        # cv_file_list = misc.create_CV_folds(PARAMS['folder'], 'HNGDMFCC', PARAMS['test_classes'], PARAMS['CV_folds'])
        # cv_file_list_test = cv_file_list
        PARAMS['test_folder'] = PARAMS['folder']
    else:
        PARAMS['test_folder'] = PARAMS['test_path']
        # cv_file_list = misc.create_CV_folds(PARAMS['folder'], 'HNGDMFCC', PARAMS['test_classes'], PARAMS['CV_folds'])
        # cv_file_list_test = misc.create_CV_folds(PARAMS['test_folder'], 'HNGDMFCC', PARAMS['test_classes'], PARAMS['CV_folds'])

    cv_file_list = {'wav':{
        'fold0': ['ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994.wav', 'ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994.wav'],
        'fold1': ['ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994.wav', 'eatmycountry1609.wav',],
        'fold2': ['theconcert16.wav', 'theconcert2.wav', 'UTMA-26.wav'],
        }
        }

    cv_file_list_test = cv_file_list


    PARAMS['opDir'] = PARAMS['test_folder'] + '/__RESULTS/' + PARAMS['today'] + '/Segmentation_' + PARAMS['segmentation_func'] + '_muspeak_standalone/'
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    PARAMS['cnn_model_path'] = PARAMS['opDir'] + '/' + PARAMS['CNN_model_type'] + '/' + PARAMS['featName'] + '/'
    if not os.path.exists(PARAMS['cnn_model_path']):
        os.makedirs(PARAMS['cnn_model_path'])
    
    misc.print_configuration(PARAMS)
    
    for feature_type in PARAMS['feat_combinations'].keys():
        feature_list = PARAMS['feat_combinations'][feature_type]
        print('\n\n\nfeature_type: ', feature_type)

        for fold in range(PARAMS['CV_folds']):

            if PARAMS['use_GPU']:
                start_GPU_session()

            PARAMS['fold'] = fold
            print('\nfoldNum: ', fold)
            train_files, test_files = misc.get_train_test_files(cv_file_list, cv_file_list, PARAMS['CV_folds'], fold)
            print('train_files: ', train_files)
            print('test_files: ', test_files)

            Sequence_Data = {'train':{}, 'test':{}}
            SP_Marker = {'train':{}, 'test':{}}
            MU_Marker = {'train':{}, 'test':{}}

            '''
            Load trained classifier models
            '''
            Ensemble_Train_Params = {}
            PARAMS['input_shape'] = {}
            for featName in feature_list:
                model_folder = PARAMS['model_folder'] + '/' + featName + '/__RESULTS/' + PARAMS['clFunc'] + '/'
                modelName = model_folder + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                if featName in ['HNGDMFCC', 'MGDCC', 'IFCC', 'MFCC-39']:
                    PARAMS['input_shape'][featName] = (39, PARAMS['CNN_patch_size'], 1)
                else:
                    PARAMS['input_shape'][featName] = (21, PARAMS['CNN_patch_size'], 1)
                Train_Params = load_model(modelName, PARAMS['input_shape'][featName])
                Ensemble_Train_Params[featName] = Train_Params
                data_stats_dir = PARAMS['cnn_model_path'] + '/__data_stats/'
                if not os.path.exists(data_stats_dir):
                    os.makedirs(data_stats_dir)
                data_stats_fName = featName + '_fold' + str(PARAMS['fold']) + '_mean_stdev'
                if not os.path.exists(data_stats_dir + '/' + data_stats_fName + '.pkl'):
                    data_mean, data_stdev = get_data_stats(PARAMS, featName, train_files)
                    misc.save_obj({'mean':data_mean,'stdev':data_stdev}, data_stats_dir, data_stats_fName)
                    print(featName, ' data stats saved')
                else:
                    data_mean = misc.load_obj(data_stats_dir, data_stats_fName)['mean']
                    data_stdev = misc.load_obj(data_stats_dir, data_stats_fName)['stdev']
                    print(featName, ' data stats loaded')

            PtdLabels_train_sp = []
            PtdLabels_train_mu = []
            GroundTruths_train_sp = []
            GroundTruths_train_mu = []
            for fl in train_files[PARAMS['test_classes'][0]]:
                Predictions_Dict_train = get_test_predictions(PARAMS, Ensemble_Train_Params, feature_type, fl)

                # print('CNN Predictions: ', np.shape(Predictions_Dict_train['Pred']), np.shape(Predictions_Dict_train['Labels_sp']), np.shape(Predictions_Dict_train['Labels_mu']))
                pred_lab_sp = np.argmax(Predictions_Dict_train['Pred'], axis=1)
                PtdLabels_train_sp.extend(pred_lab_sp.tolist())
                GroundTruths_train_sp.extend(Predictions_Dict_train['Labels_sp'].flatten())
                pred_lab_mu = np.argmin(Predictions_Dict_train['Pred'], axis=1)
                PtdLabels_train_mu.extend(pred_lab_mu.tolist())
                GroundTruths_train_mu.extend(Predictions_Dict_train['Labels_mu'].flatten())
                ConfMat_sp, fscore_sp = getPerformance(pred_lab_sp, Predictions_Dict_train['Labels_sp'])
                ConfMat_mu, fscore_mu = getPerformance(pred_lab_mu, Predictions_Dict_train['Labels_mu'])
                                
                Sequence_Data['train'][fl] = Predictions_Dict_train['Pred']
                SP_Marker['train'][fl] = Predictions_Dict_train['Labels_sp']
                MU_Marker['train'][fl] = Predictions_Dict_train['Labels_mu']
                # print(fl, ' CNN predictions train: ', np.shape(Predictions_Dict_train['Pred']), np.shape(Predictions_Dict_train['Labels_sp']), np.shape(Predictions_Dict_train['Labels_mu']))
                Predictions_Dict_train = None
                del Predictions_Dict_train
            
            print('Training all files sp: ', np.shape(PtdLabels_train_sp), np.shape(GroundTruths_train_sp))
            print('Training all files mu: ', np.shape(PtdLabels_train_mu), np.shape(GroundTruths_train_mu))

            GroundTruths_train_sp = np.array(GroundTruths_train_sp)
            GroundTruths_train_mu = np.array(GroundTruths_train_mu)
            PtdLabels_train_sp = np.array(PtdLabels_train_sp)
            PtdLabels_train_mu = np.array(PtdLabels_train_mu)

            ConfMat_sp, fscore_sp = getPerformance(PtdLabels_train_sp, GroundTruths_train_sp)
            print('Total training data performance sp: ', fscore_sp)
            ConfMat_mu, fscore_mu = getPerformance(PtdLabels_train_mu, GroundTruths_train_mu)
            print('Total training data performance mu: ', fscore_mu)

            kwargs = {
                '0':'data:train',
                '1':'feature_type:'+feature_type,
                '2':'F_score_nsp:'+str(fscore_sp[0]),
                '3':'F_score_sp:'+str(fscore_sp[1]),
                '4':'F_score_sp_avg:'+str(fscore_sp[2]),
                '5':'F_score_nmu:'+str(fscore_mu[0]),
                '6':'F_score_mu:'+str(fscore_mu[1]),
                '7':'F_score_mu_avg:'+str(fscore_mu[2]),
                }
            misc.print_results(PARAMS, PARAMS['cnn_model_path'], PARAMS['CNN_model_type'][2:], **kwargs)
            fscore_mu = None
            fscore_sp = None

            PtdLabels_test_sp = []
            PtdLabels_test_mu = []
            GroundTruths_test_sp = []
            GroundTruths_test_mu = []
            for fl in test_files[PARAMS['test_classes'][0]]:
                Predictions_Dict_test = get_test_predictions(PARAMS, Ensemble_Train_Params, feature_type, fl)

                # print('CNN Predictions: ', np.shape(Predictions_Dict_test['Pred']), np.shape(Predictions_Dict_test['Labels_sp']), np.shape(Predictions_Dict_test['Labels_mu']))
                pred_lab_sp = np.argmax(Predictions_Dict_test['Pred'], axis=1)
                PtdLabels_test_sp.extend(pred_lab_sp.tolist())
                GroundTruths_test_sp.extend(Predictions_Dict_test['Labels_sp'].flatten())
                pred_lab_mu = np.argmin(Predictions_Dict_test['Pred'], axis=1)
                PtdLabels_test_mu.extend(pred_lab_mu.tolist())
                GroundTruths_test_mu.extend(Predictions_Dict_test['Labels_mu'].flatten())
                ConfMat_sp, fscore_sp = getPerformance(pred_lab_sp, Predictions_Dict_test['Labels_sp'])
                ConfMat_mu, fscore_mu = getPerformance(pred_lab_mu, Predictions_Dict_test['Labels_mu'])

                Sequence_Data['test'][fl] = Predictions_Dict_test['Pred']
                SP_Marker['test'][fl] = Predictions_Dict_test['Labels_sp']
                MU_Marker['test'][fl] = Predictions_Dict_test['Labels_mu']
                # print(fl, ' CNN predictions test: ', np.shape(Predictions_Dict_test['Pred']), np.shape(Predictions_Dict_test['Labels_sp']), np.shape(Predictions_Dict_test['Labels_mu']))
                Predictions_Dict_test = None
                del Predictions_Dict_test

            print('Testing all files sp: ', np.shape(PtdLabels_test_sp), np.shape(GroundTruths_test_sp))
            print('Testing all files mu: ', np.shape(PtdLabels_test_mu), np.shape(GroundTruths_test_mu))
            
            GroundTruths_test_sp = np.array(GroundTruths_test_sp)
            GroundTruths_test_mu = np.array(GroundTruths_test_mu)
            PtdLabels_test_sp = np.array(PtdLabels_test_sp)
            PtdLabels_test_mu = np.array(PtdLabels_test_mu)

            ConfMat_sp, fscore_sp = getPerformance(PtdLabels_test_sp, GroundTruths_test_sp)
            print('Total testing data performance sp: ', fscore_sp)
            ConfMat_mu, fscore_mu = getPerformance(PtdLabels_test_mu, GroundTruths_test_mu)
            print('Total testing data performance mu: ', fscore_mu)

            kwargs = {
                '0':'data:test',
                '1':'feature_type:'+feature_type,
                '2':'F_score_nsp:'+str(fscore_sp[0]),
                '3':'F_score_sp:'+str(fscore_sp[1]),
                '4':'F_score_sp_avg:'+str(fscore_sp[2]),
                '5':'F_score_nmu:'+str(fscore_mu[0]),
                '6':'F_score_mu:'+str(fscore_mu[1]),
                '7':'F_score_mu_avg:'+str(fscore_mu[2]),
                }
            misc.print_results(PARAMS, PARAMS['cnn_model_path'], PARAMS['CNN_model_type'][2:], **kwargs)

            # continue

            #%%
            #'
            # Learning segmentation model
            #'
            train_seg_data, train_seg_label = get_segmentation_data(Sequence_Data, SP_Marker, MU_Marker, 'train', train_files[PARAMS['test_classes'][0]], win_size=PARAMS['smoothing_win_size']) #PARAMS['smoothing_win_size']
            print('HMM training data: ', np.shape(train_seg_data), np.shape(train_seg_label))

            # fig_opDir = PARAMS['cnn_model_path']+'/__histograms/'
            # if not os.path.exists(fig_opDir):
            #     os.makedirs(fig_opDir)
            # plot_data_histogram(train_seg_data, train_seg_label, fig_opDir)       
            # continue
            



            hmm_model_path = PARAMS['cnn_model_path'] + '/__HMM_Models/' + feature_type + '/'
            if not os.path.exists(hmm_model_path):
                os.makedirs(hmm_model_path)
            PARAMS['modelName'] = hmm_model_path + '/fold' + str(PARAMS['fold']) + '.xyz'
            nMix = 2
            HMM_Train_PARAMS = learn_hmm(PARAMS, nMix, train_seg_data, train_seg_label)



            
            Test_Params = {}
            test_params_fName = 'HMM_Test_Params_fold' + str(fold) + '_' + feature_type
            if not os.path.exists(hmm_model_path+'/'+test_params_fName+'.pkl'):
                test_seg_data, test_seg_label = get_segmentation_data(Sequence_Data, SP_Marker, MU_Marker, 'test', test_files[PARAMS['test_classes'][0]], win_size=PARAMS['smoothing_win_size'])
                
                Test_Params = test_hmm(PARAMS, HMM_Train_PARAMS, test_seg_data, test_seg_label)
                
                for fl in test_files[PARAMS['test_classes'][0]]:
                    test_seg_data, test_seg_label = get_segmentation_data(Sequence_Data, SP_Marker, MU_Marker, 'test', [fl], win_size=PARAMS['smoothing_win_size'])
                    if np.size(test_seg_data)<=1:
                        continue
                    Test_Params[fl] = test_hmm(PARAMS, HMM_Train_PARAMS, test_seg_data, test_seg_label)
                    print('\nTest results', fl, np.shape(test_seg_data), np.shape(Test_Params[fl]['PtdLabels_test']), np.shape(Sequence_Data['test'][fl]), np.shape(SP_Marker['test'][fl]))
                misc.save_obj(Test_Params, hmm_model_path, test_params_fName)
            else:
                Test_Params = misc.load_obj(hmm_model_path, test_params_fName)
            # print('Test_Params: ', Test_Params.keys())                

            kwargs = {
                '0': 'nMix:' + str(nMix),
                '1':'training_time:'+str(HMM_Train_PARAMS['trainingTimeTaken']),
                '2':'Accuracy:'+str(Test_Params['Accuracy']),
                '3':'Precision:'+str(Test_Params['precision']),
                '4':'Recall:'+str(Test_Params['recall']),
                '5':'F_score_mu:'+str(Test_Params['fscore'][0]),
                '6':'F_score_sp:'+str(Test_Params['fscore'][1]),
                '7':'F_score_avg:'+str(Test_Params['fscore'][2]),
                }
            misc.print_results(PARAMS, PARAMS['cnn_model_path'], '', **kwargs)
    
    
    
    
            for fl in test_files[PARAMS['test_classes'][0]]:
                if not fl in Test_Params.keys():
                    continue
                print('\n\n\n', fl, ' fscore: ', Test_Params[fl]['fscore'])
   
                #'' Plot segmentation results ''
                PARAMS['plot_fig'] = False
                if PARAMS['plot_fig']:
                    opDirFig = PARAMS['cnn_model_path'] + '/__figures/' + feature_type + '/fold' + str(PARAMS['fold']) + '/'
                    # print(opDirFig, os.path.exists(opDirFig))
                    if not os.path.exists(opDirFig):
                        os.makedirs(opDirFig)
                    # print('Plot data ', fl, np.shape(Sequence_Data['test'][fl]), np.shape(Test_Params[fl]['PtdLabels_test']), np.shape(SP_Marker['test'][fl]), np.shape(MU_Marker['test'][fl]))
                    plot_segmentation_results_hmm(PARAMS, opDirFig, Sequence_Data['test'][fl], Test_Params[fl]['PtdLabels_test'], PARAMS['annot_path'], fl, SP_Marker['test'][fl], MU_Marker['test'][fl], win_size=PARAMS['smoothing_win_size'])
                    
                seg_opDir = PARAMS['cnn_model_path'] + '/__segmentation_results/' + feature_type + '/fold' + str(PARAMS['fold']) + '/'
                if not os.path.exists(seg_opDir):
                    os.makedirs(seg_opDir)
                if not os.path.exists(seg_opDir+'/'+fl.split('.')[0]+'.mat'):
                    seg_results = {
                        'fl': fl,
                        'CNN_Predictions': Sequence_Data['test'][fl],
                        'speech_groundtruth': SP_Marker['test'][fl],
                        'music_groundtruth': MU_Marker['test'][fl],
                        'HMM_PtdLabels': Test_Params[fl]['PtdLabels_test'],
                        }
                    savemat(seg_opDir+'/'+fl.split('.')[0]+'.mat', seg_results)
            
            print('Segmentation results computing')
            print_segmentation_results([5000], PARAMS['smoothing_win_size'], PARAMS['cnn_model_path'], 'onoff', feature_type, fold, plot_fig=False, segmentation_track=False, segFunc='HMM')
            print_segmentation_results([1000], PARAMS['smoothing_win_size'], PARAMS['cnn_model_path'], 'onoff', feature_type, fold, plot_fig=False, segmentation_track=False, segFunc='HMM')
            print_segmentation_results([500], PARAMS['smoothing_win_size'], PARAMS['cnn_model_path'], 'onoff', feature_type, fold, plot_fig=False, segmentation_track=False, segFunc='HMM')
            
            if PARAMS['use_GPU']:
                reset_TF_session()
