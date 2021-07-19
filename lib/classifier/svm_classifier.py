#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:33:11 2019
Updated on Thu Jun 6 2019
Updated on Tue Apr  13 17:15:23 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.model_selection import GridSearchCV
import multiprocessing
import lib.misc as misc
import os
import time
import scipy





def grid_search_svm(PARAMS, train_data, train_label, test_data, test_label):
    C = np.power(2.0, list(np.arange(-5,5,1)))
    Gamma = np.power(2.0, list(np.arange(-5,5,1)))
    njobs = multiprocessing.cpu_count() - 1
    cv_folds = 3
    print('CV folds=', cv_folds, ' n_jobs=', njobs)

    trainingTimeTaken = 0
    start = time.process_time()
    
    clf_param_tuning = SVC(decision_function_shape='ovo', verbose=0, probability=True)
    tunable_parameters = [{'kernel': ['rbf'], 'gamma': Gamma, 'C': C}]
    CLF_CV = GridSearchCV(clf_param_tuning, tunable_parameters, cv=cv_folds, refit=True, n_jobs=njobs, verbose=2)

    '''
    Checking if model is already available
    '''
    if not os.path.exists(PARAMS['modelName']):
        CLF_CV.fit(train_data, train_label.flatten())
        model = CLF_CV.best_estimator_
        if PARAMS['save_flag']:
            misc.save_obj(model, PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0])
            misc.save_obj(CLF_CV, PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_All_Models')
    else:
        model = misc.load_obj(PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0])
        CLF_CV = misc.load_obj(PARAMS['opDir'], PARAMS['modelName'].split('/')[-1].split('.')[0]+'_All_Models')

    trainingTimeTaken = time.process_time() - start
    
    testingTimeTaken = 0
    start = time.process_time()

    optC = str(CLF_CV.best_params_['C'])
    optGamma = str(CLF_CV.best_params_['gamma'])
    countSV = model.n_support_
        
    countTrPts = [np.sum(train_label==lab) for lab in np.unique(train_label)]

    PtdLabels_train = model.predict(train_data)
    # Predictions_train = model.predict_log_proba(train_data)

    PtdLabels_test = model.predict(test_data)
    Predictions_test = model.predict_log_proba(test_data)

    accuracy_train = np.mean(PtdLabels_train.ravel() == train_label.ravel()) * 100
    accuracy_test = np.mean(PtdLabels_test.ravel() == test_label.ravel()) * 100
    
    ConfMat_train, fscore_train = misc.getPerformance(PtdLabels_train, train_label)
    ConfMat_test, fscore_test = misc.getPerformance(PtdLabels_test, test_label)
    
    # Performance_train = np.array([accuracy_train, fscore_train[0], fscore_train[1], fscore_train[2]])
    Performance_test = np.array([accuracy_test, fscore_test[0], fscore_test[1], fscore_test[2]])
    
    print('Accuracy: train=', accuracy_train, ' test=', accuracy_test, 'F-score: train=', fscore_train[-1], ' test=', fscore_test[-1], ' SupportVectors=', countSV)
    testingTimeTaken = time.process_time() - start
    
    
    Train_Params = {
        'model':model,
        'optC': optC,
        'optGamma': optGamma,
        'countSV': countSV,
        'countTrPts': countTrPts,
        'trainingTimeTaken': trainingTimeTaken,
        }
    
    Test_Params = {
        'PtdLabels': PtdLabels_test,
        'Predictions': Predictions_test,
        'accuracy': accuracy_test,
        'Performance_test': Performance_test,
        'testingTimeTaken': testingTimeTaken,
        'fscore': fscore_test,
        'GroundTruth': test_label,
        }

    # return model, optC, optGamma, Predictions_test, Performance_test, countSV, countTrPts
    return Train_Params, Test_Params





def test_model_svm(PARAMS, test_data, test_label, input_shape):
    '''
    Checking if model is already available
    '''
    if not os.path.exists(PARAMS['modelName']):
        print('Model does not exist')
        return {}, {}
    else:
        model = misc.load_obj('/'.join(PARAMS['modelName'].split('/')[:-1]), PARAMS['modelName'].split('/')[-1].split('.')[0])
        CLF_CV = misc.load_obj('/'.join(PARAMS['modelName'].split('/')[:-1]), PARAMS['modelName'].split('/')[-1].split('.')[0]+'_All_Models')

    testingTimeTaken = 0
    start = time.process_time()

    optC = str(CLF_CV.best_params_['C'])
    optGamma = str(CLF_CV.best_params_['gamma'])
    countSV = model.n_support_

    # PtdLabels_test = model.predict(test_data)
    # Predictions_test = model.predict_log_proba(test_data)
    # GroundTruth = test_label
    
    temp_folder = PARAMS['opDir'] + '/__temp/'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    PtdLabels_test = []
    Predictions_test = []
    GroundTruth = []
    for clNum in PARAMS['classes'].keys():
        files = PARAMS['test_files'][PARAMS['classes'][clNum]]
        for fl in files:
            temp_file = 'pred_' + PARAMS['classes'][clNum] + '_fold' + str(PARAMS['fold']) + '_' + PARAMS['featName'] + '_' + fl.split('.')[0]
            if not os.path.exists(temp_folder + temp_file + '.pkl'):
                FV = np.load(PARAMS['test_folder'] + '/' + PARAMS['featName'] + '/' + PARAMS['classes'][clNum] + '/' + fl, allow_pickle=True)
                FV = misc.get_feature_patches(FV, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], input_shape)
                FV = PARAMS['std_scale'].transform(FV)
                pred_lab = model.predict(FV)
                pred = model.predict_proba(FV)
                misc.save_obj({'pred':pred, 'pred_lab':pred_lab}, temp_folder, temp_file)
            else:
                pred = misc.load_obj(temp_folder, temp_file)['pred']
                pred_lab = misc.load_obj(temp_folder, temp_file)['pred_lab']
            PtdLabels_test.extend(pred_lab)
            Predictions_test.extend(pred)
            GroundTruth.extend([clNum]*np.shape(pred)[0])
            print(fl, ' acc=', np.sum(np.array(pred_lab)==np.array([clNum]*np.shape(pred)[0]))/np.size(pred_lab))
    PtdLabels_test = np.array(PtdLabels_test)
    GroundTruth = np.array(GroundTruth)
            
    accuracy_test = np.mean(PtdLabels_test.ravel() == GroundTruth.ravel()) * 100
    ConfMat_test, fscore_test = misc.getPerformance(PtdLabels_test, GroundTruth)
    Performance_test = np.array([accuracy_test, fscore_test[0], fscore_test[1], fscore_test[2]])
    testingTimeTaken = time.process_time() - start    
    print('Accuracy: test=', np.round(accuracy_test,4), 'F-score: test=', np.round(fscore_test,4))
        
    Train_Params = {
        'model':model,
        'optC': optC,
        'optGamma': optGamma,
        }
    
    Test_Params = {
        'PtdLabels': PtdLabels_test,
        'Predictions': Predictions_test,
        'accuracy': accuracy_test,
        'Performance_test': Performance_test,
        'testingTimeTaken': testingTimeTaken,
        'fscore': fscore_test,
        'GroundTruth': GroundTruth,
        }

    return Train_Params, Test_Params



def test_svm_ensemble(PARAMS, All_Test_Params):
    PtdLabels_Ensemble = []
    GroundTruth_Ensemble = []
    Predictions_Ensemble = np.empty([])
    featCount = 0
    testingTimeTaken = 0
    start = time.process_time()
    for featName in All_Test_Params.keys():
        if not 'PtdLabels' in All_Test_Params[featName].keys():
            continue
        Test_Params = All_Test_Params[featName]
        if featCount==0:
            Predictions_Ensemble = np.array(Test_Params['PtdLabels'], ndmin=2).T
            GroundTruth_Ensemble = Test_Params['GroundTruth']
        else:
            Predictions_Ensemble = np.append(Predictions_Ensemble, np.array(Test_Params['PtdLabels'], ndmin=2).T, axis=1)
        featCount += 1
    PtdLabels_Ensemble, mode_count = scipy.stats.mode(Predictions_Ensemble, axis=1)
    ConfMat_Ensemble, fscore_Ensemble = misc.getPerformance(PtdLabels_Ensemble, GroundTruth_Ensemble)
    accuracy_Ensemble = np.sum(np.diag(ConfMat_Ensemble))/np.sum(ConfMat_Ensemble)
    testingTimeTaken = time.process_time() - start
    print('SVM Ensemble: ', accuracy_Ensemble, fscore_Ensemble)

    Ensemble_Test_Params = {
        'accuracy_Ensemble': accuracy_Ensemble,
        'testingTimeTaken': testingTimeTaken,
        'ConfMat_Ensemble': ConfMat_Ensemble,
        'fscore_Ensemble': fscore_Ensemble,
        'PtdLabels_Ensemble': PtdLabels_Ensemble,
        'Predictions_Ensemble': Predictions_Ensemble,
        'GroundTruth_Ensemble': GroundTruth_Ensemble,
        }
    
    return Ensemble_Test_Params