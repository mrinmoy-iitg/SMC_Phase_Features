#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:34:47 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import configparser
import datetime
import numpy as np
import lib.misc as misc
import lib.classifier.dnn_classifier as DNN
import lib.classifier.cnn_classifier as CNN
import lib.classifier.svm_classifier as SVM
import lib.classifier.NB_classifier as NB
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler





def start_GPU_session():
    tf.keras.backend.clear_session()
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
    tf.compat.v1.keras.backend.clear_session()



def array_mode(X, axis=1):
    labels = []
    if axis==1:
        for i in range(np.shape(X)[0]):
            # print(i, np.shape(X))
            count = [0,0]
            for lab in X[i, :]:
                count[lab] += 1
            labels.append(np.argmax(count))
    else:
        for i in range(np.shape(X)[1]):
            count = [0,0]
            for lab in X[:, i]:
                count[lab] += 1
            labels.append(np.argmax(count))

    return labels



def load_model(PARAMS, clFunc):
    modelName = '@'.join(PARAMS['modelName'].split('.')[:-1]) + '.' + PARAMS['modelName'].split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
        
    if os.path.exists(paramFile):
        if (clFunc=='CNN') or (clFunc=='CNN_TL'):
            epochs = np.load(paramFile)['epochs']
            batch_size = np.load(paramFile)['batch_size']
            learning_rate = float(np.load(paramFile)['lr'])
            trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        else:
            epochs = np.load(paramFile)['ep']
            batch_size = np.load(paramFile)['bs']
            learning_rate = float(np.load(paramFile)['lr'])
            trainingTimeTaken = np.load(paramFile)['TTT']
        
        # print(epochs, batch_size, learning_rate, trainingTimeTaken)
        optimizer = Adam(lr=learning_rate)
        try:
            with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
                model = model_from_json(f.read())
        except:
            if (clFunc=='CNN') or (clFunc=='CNN_TL'):
                model, learning_rate_temp = CNN.get_cnn_model(PARAMS['input_shape'][PARAMS['featName']], 2)
            elif clFunc=='DNN':
                num_dnn_lyr = PARAMS['DNN_optimal_params'][PARAMS['dataset_name']][PARAMS['featName']]['hidden_lyrs']
                num_dnn_nodes = PARAMS['DNN_optimal_params'][PARAMS['dataset_name']][PARAMS['featName']]['hidden_nodes']
                model, optimizerName, learning_rate_temp = DNN.dnn_model(PARAMS['input_dim'], 1, num_dnn_lyr, num_dnn_nodes)
                
        model.load_weights(weightFile) # Load weights into the new model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('\n\t\t\t', clFunc, ' model exists! Loaded. Training time required=',trainingTimeTaken)

        Train_Params = {
                'model': model,
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
        



'''
Initialize the script
'''
def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Ensemble_Classifier.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': section['folder'],
            'test_path': section['test_path'],
            'test_folder': '',
            'clFunc': section['clFunc'], #DNN-Ensemble, CNN-Ensemble, CNN-TL-Ensemble, SVM-Ensemble, NB-Ensemble
            'dataset_name': section['dataset_name'],
            'noise_experiment': section.getboolean('noise_experiment'),
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'epochs': int(section['epochs']),
            'batch_size': int(section['batch_size']),
            'CV_folds': int(section['CV_folds']),
            'fold': 0,
            'save_flag': section.getboolean('save_flag'),
            'data_generator': section.getboolean('data_generator'),
            'data_balancing': section.getboolean('data_balancing'),
            'use_GPU': section.getboolean('use_GPU'),
            'train_steps_per_epoch':0,
            'val_steps':0,
            'scale_data': section.getboolean('scale_data'),
            'PCA_flag': section.getboolean('PCA_flag'),
            'GPU_session':None,
            'output_folder':'',
            'classes':{0:'music', 1:'speech'},
            'dataset':'',
            'feat_type':'',
            'opDir':'',
            'modelName':'',
            'input_dim':0,
            'feat_combinations':{
                'all_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'],
                'magnitude_features': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram'],
                'phase_features': ['HNGDMFCC', 'MGDCC', 'IFCC'],
                'magnitude_features+HNGDMFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC'],
                'magnitude_features+MGDCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'MGDCC'],
                'magnitude_features+IFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'IFCC'],
                'all_features-HNGDMFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'MGDCC', 'IFCC'],
                'all_features-MGDCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'IFCC'],
                'all_features-IFCC': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC'],
                'HNGDMFCC+MGDCC': ['HNGDMFCC', 'MGDCC'],
                'HNGDMFCC+IFCC': ['HNGDMFCC', 'IFCC'],
                'MGDCC+IFCC': ['MGDCC', 'IFCC'],
                'Khonglah_et_al+HNGDMFCC': ['Khonglah_et_al', 'HNGDMFCC'],
                'Khonglah_et_al+MGDCC': ['Khonglah_et_al', 'MGDCC'],
                'Khonglah_et_al+IFCC': ['Khonglah_et_al', 'IFCC'],
                'Sell_et_al+HNGDMFCC': ['Sell_et_al', 'HNGDMFCC'],
                'Sell_et_al+MGDCC': ['Sell_et_al', 'MGDCC'],
                'Sell_et_al+IFCC': ['Sell_et_al', 'IFCC'],
                'MFCC-39+HNGDMFCC': ['MFCC-39', 'HNGDMFCC'],
                'MFCC-39+MGDCC': ['MFCC-39', 'MGDCC'],
                'MFCC-39+IFCC': ['MFCC-39', 'IFCC'],
                'Melspectrogram+HNGDMFCC': ['Melspectrogram', 'HNGDMFCC'],
                'Melspectrogram+MGDCC': ['Melspectrogram', 'MGDCC'],
                'Melspectrogram+IFCC': ['Melspectrogram', 'IFCC'],
                }, #['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'HNGDMFCC', 'MGDCC', 'IFCC', 'Melspectrogram']
            'featName': '',
            '39_dim_CC_feat': section.getboolean('39_dim_CC_feat'),
            'noise_dB_range': [10, 8, 5, 2, 1, 0],
            'DNN_optimal_params': {
                'MUSAN':
                {'Khonglah_et_al':{'hidden_lyrs':3, 'hidden_nodes':250},
                'Sell_et_al':{'hidden_lyrs':2, 'hidden_nodes':100},
                'MFCC-39':{'hidden_lyrs':3, 'hidden_nodes':100}, #updated
                'Melspectrogram':{'hidden_lyrs':5, 'hidden_nodes':250},
                'HNGDMFCC':{'hidden_lyrs':5, 'hidden_nodes':100}, #updated
                'MGDCC':{'hidden_lyrs':4, 'hidden_nodes':500},
                'IFCC':{'hidden_lyrs':4, 'hidden_nodes':100},},
                'MUSNOMIX_WAV':
                {'Khonglah_et_al':{'hidden_lyrs':5, 'hidden_nodes':250}, #updated
                'Sell_et_al':{'hidden_lyrs':3, 'hidden_nodes':1000}, #updated
                'MFCC-39':{'hidden_lyrs':4, 'hidden_nodes':500}, #updated
                'Melspectrogram':{'hidden_lyrs':2, 'hidden_nodes':1000}, #updated
                'HNGDMFCC':{'hidden_lyrs':3, 'hidden_nodes':500}, #updated
                'MGDCC':{'hidden_lyrs':2, 'hidden_nodes':1000}, #updated
                'IFCC':{'hidden_lyrs':5, 'hidden_nodes':500}} #updated
                },
            }
   
    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
        
    for feature_type in PARAMS['feat_combinations'].keys():
        feature_list = PARAMS['feat_combinations'][feature_type]
        print('\n\n\nfeature_type: ', feature_type)

        if PARAMS['use_GPU']:
            start_GPU_session()

        for foldNum in range(PARAMS['CV_folds']):
            PARAMS['fold'] = foldNum
            print('\nfoldNum: ', foldNum)
            
            Ensemble_GroundTruths = np.empty([])
            Ensemble_PtdLabels = np.empty([])
            Ensemble_Predictions = np.empty([])
            PtdLabels_majority_voting = np.empty([])
            Ensemble_Train_Params = {}
            All_Test_Params = {'Khonglah_et_al':{'fscore': [0, 0, 0]}, 'Sell_et_al':{'fscore': [0, 0, 0]}, 'MFCC-39':{'fscore': [0, 0, 0]}, 'Melspectrogram':{'fscore': [0, 0, 0]}, 'HNGDMFCC':{'fscore': [0, 0, 0]}, 'MGDCC':{'fscore': [0, 0, 0]}, 'IFCC':{'fscore': [0, 0, 0]}}
            
            PARAMS['input_shape'] = {}
            for PARAMS['featName'] in feature_list:
                if PARAMS['featName']=='Melspectrogram':
                    PARAMS['CNN_feat_dim'] = 21
                    PARAMS['input_dim'] = 42
                elif (PARAMS['featName']=='HNGDMFCC') or (PARAMS['featName']=='MGDCC') or (PARAMS['featName']=='IFCC') or (PARAMS['featName']=='MFCC-39'):
                    PARAMS['CNN_feat_dim'] = 21
                    PARAMS['input_dim'] = 42
                elif PARAMS['featName']=='Khonglah_et_al':
                    PARAMS['CNN_feat_dim'] = 10
                    PARAMS['input_dim'] = 20
                elif PARAMS['featName']=='Sell_et_al':
                    PARAMS['CNN_feat_dim'] = 9
                    PARAMS['input_dim'] = 18
                # print(PARAMS['input_shape'])
                PARAMS['input_shape'][PARAMS['featName']] = (21, PARAMS['CNN_patch_size'], 1)
                
                if PARAMS['39_dim_CC_feat']:
                    if (PARAMS['featName']=='HNGDMFCC') or (PARAMS['featName']=='MGDCC') or (PARAMS['featName']=='IFCC') or (PARAMS['featName']=='MFCC-39'):
                        PARAMS['CNN_feat_dim'] = 39
                        PARAMS['input_dim'] = 78
                        PARAMS['input_shape'][PARAMS['featName']] = (39, PARAMS['CNN_patch_size'], 1)
                        
                # print(PARAMS['featName'], PARAMS['input_shape'][PARAMS['featName']])

                '''
                Initializations
                '''
                opDir_suffix = ''
                if PARAMS['test_path']=='':
                    cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
                    cv_file_list_test = cv_file_list
                    PARAMS['test_folder'] = PARAMS['folder']
                    PARAMS['output_folder'] = PARAMS['test_folder'] + '/__RESULTS/' + PARAMS['today'] + '/'
                else:
                    PARAMS['test_folder'] = PARAMS['test_path']
                    cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
                    cv_file_list_test = misc.create_CV_folds(PARAMS['test_folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
                    PARAMS['output_folder'] = PARAMS['test_folder'] + '/__RESULTS/' + PARAMS['today'] + '/'
                    opDir_suffix = '_GEN_PERF_' + PARAMS['folder'].split('/')[-3]
                    
                if not os.path.exists(PARAMS['output_folder']):
                    os.makedirs(PARAMS['output_folder'])
        
                PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['clFunc'] + opDir_suffix + '/'
                print('opDir: ', PARAMS['opDir'] )
                if not os.path.exists(PARAMS['opDir']):
                    os.makedirs(PARAMS['opDir'])
                
                misc.print_configuration(PARAMS)

                PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)


                '''
                Load data
                '''
                if not PARAMS['data_generator']:
                    train_data, train_label = misc.load_data_from_files(PARAMS['classes'], PARAMS['folder'], PARAMS['featName'], PARAMS['train_files'], PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'][PARAMS['featName']])
                    PARAMS['std_scale'] = StandardScaler().fit(train_data)
                    test_data, test_label = misc.load_data_from_files(PARAMS['classes'], PARAMS['test_folder'], PARAMS['featName'], PARAMS['test_files'], PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'][PARAMS['featName']])
                    print('Data loaded: ', np.shape(train_data), np.shape(train_label), np.shape(test_data), np.shape(test_label))
                    
                    train_data, train_label, test_data = misc.preprocess_data(PARAMS, train_data, train_label, test_data)
                    print('Data preprocessed: ', np.shape(train_data), np.shape(test_data))
                else:
                    train_data = np.empty([])
                    train_label = np.empty([])
                    test_data = np.empty([])
                    test_label = np.empty([])

                                            
                if PARAMS['clFunc']=='DNN-Ensemble':
                    '''
                    Trained models for each feature set must be kept at:
                        PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/DNN/' 
                    '''
                    model_folder = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/DNN/'
                    PARAMS['modelName'] = model_folder + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                    # PARAMS['input_shape'] = None

                    Train_Params = load_model(PARAMS, 'DNN')
                    if len(Train_Params)==0:
                        print('featName: ', PARAMS['featName'])
                        import sys
                        sys.exit(0)
                    Ensemble_Train_Params[PARAMS['featName']] = Train_Params

                elif PARAMS['clFunc']=='CNN-Ensemble':                        
                    '''
                    Trained models for each feature set must be kept at:
                        PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/CNN/' 
                    '''
                    model_folder = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/CNN/'
                    PARAMS['modelName'] = model_folder + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                    # PARAMS['input_shape'] = (21, PARAMS['CNN_patch_size'], 1)

                    Train_Params = None
                    Train_Params = load_model(PARAMS, 'CNN')
                    if len(Train_Params)==0:
                        print('featName: ', PARAMS['featName'])
                        import sys
                        sys.exit(0)
                    Ensemble_Train_Params[PARAMS['featName']] = Train_Params

                elif PARAMS['clFunc']=='CNN-TL-Ensemble':
                    '''
                    Trained models for each feature set must be kept at:
                        PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/CNN_TL/' 
                    '''
                    model_folder = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/CNN_TL/'
                    PARAMS['modelName'] = model_folder + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                    # PARAMS['input_shape'] = (21, PARAMS['CNN_patch_size'], 1)

                    Train_Params = None
                    Train_Params = load_model(PARAMS, 'CNN_TL')
                    if len(Train_Params)==0:
                        print('featName: ', PARAMS['featName'])
                        import sys
                        sys.exit(0)
                    Ensemble_Train_Params[PARAMS['featName']] = Train_Params


                elif PARAMS['clFunc']=='SVM-Ensemble':
                    '''
                    Trained models for each feature set must be kept at:
                        PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/SVM/' 
                    '''
                    model_folder = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/SVM/'
                    PARAMS['modelName'] = model_folder + '/fold' + str(PARAMS['fold']) + '_model.pkl'
                    
                    Train_Params = None
                    Test_Params = None
                    Train_Params, Test_Params  = SVM.test_model_svm(PARAMS, test_data, test_label, PARAMS['input_shape'][PARAMS['featName']])
                    if len(Train_Params)==0:
                        print('featName: ', PARAMS['featName'])
                        import sys
                        sys.exit(0)
                    Ensemble_Train_Params[PARAMS['featName']] = Train_Params
                    All_Test_Params[PARAMS['featName']] = Test_Params


                elif PARAMS['clFunc']=='NB-Ensemble':
                    '''
                    Trained models for each feature set must be kept at:
                        PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/SVM/' 
                    '''
                    model_folder = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/NB/'
                    PARAMS['modelName'] = model_folder + '/fold' + str(PARAMS['fold']) + '_model.pkl'
                    
                    Train_Params = None
                    Test_Params = None
                    Train_Params, Test_Params  = NB.load_model_NB(PARAMS, test_data, test_label, PARAMS['input_shape'][PARAMS['featName']])
                    if len(Train_Params)==0:
                        print('featName: ', PARAMS['featName'])
                        import sys
                        sys.exit(0)
                    Ensemble_Train_Params[PARAMS['featName']] = Train_Params
                    All_Test_Params[PARAMS['featName']] = Test_Params

            
            if not PARAMS['noise_experiment']:            
                if  PARAMS['clFunc']=='DNN-Ensemble':
                    Ensemble_Test_Params = DNN.test_dnn_ensemble(PARAMS, Ensemble_Train_Params)
                elif (PARAMS['clFunc']=='CNN-Ensemble') or (PARAMS['clFunc']=='CNN-TL-Ensemble'):
                    Ensemble_Test_Params = CNN.test_cnn_ensemble(PARAMS, Ensemble_Train_Params)
                elif  PARAMS['clFunc']=='SVM-Ensemble':
                    Ensemble_Test_Params = SVM.test_svm_ensemble(PARAMS, All_Test_Params)
                elif  PARAMS['clFunc']=='NB-Ensemble':
                    Ensemble_Test_Params = NB.test_NB_ensemble(PARAMS, All_Test_Params)
    
                print(feature_type, foldNum, ' Ensemble Avg. F1-score: ', Ensemble_Test_Params['fscore_Ensemble'][2])
                
                resultFile = PARAMS['opDir'] + '/Ensemble_performance_' + feature_type + '.csv'
                result_fid = open(resultFile, 'a+', encoding='utf-8')
                result_fid.write('Majority Voting Ensemble Average\t' +
                                 str(Ensemble_Test_Params['fscore_Ensemble'][0]) + '\t' +
                                 str(Ensemble_Test_Params['fscore_Ensemble'][1]) + '\t' +
                                 str(Ensemble_Test_Params['fscore_Ensemble'][2]) + '\n')
                result_fid.close()
                
                kwargs = {}
                kwargs['0'] = 'feature_type:'+feature_type
                i = 0
                if  (PARAMS['clFunc']=='DNN-Ensemble') or (PARAMS['clFunc']=='CNN-Ensemble'):
                    for key in Ensemble_Test_Params['individual_performances'].keys():
                        kwargs[str(i+1)] = key + ':' + str(Ensemble_Test_Params['individual_performances'][key]['fscore'][2])
                        i += 1
                elif  (PARAMS['clFunc']=='SVM-Ensemble') or (PARAMS['clFunc']=='NB-Ensemble'):
                    for key in All_Test_Params.keys():
                        kwargs[str(i+1)] = key + ':' + str(All_Test_Params[key]['fscore'][2])
                        i += 1
                    
                kwargs['8'] = 'Accuracy:'+str(Ensemble_Test_Params['accuracy_Ensemble'])
                kwargs['9'] = 'F_score_mu:'+str(Ensemble_Test_Params['fscore_Ensemble'][0])
                kwargs['10'] = 'F_score_sp:'+str(Ensemble_Test_Params['fscore_Ensemble'][1])
                kwargs['11'] = 'F_score_avg:'+str(Ensemble_Test_Params['fscore_Ensemble'][2])
                
                misc.print_results(PARAMS, '', **kwargs)
            else:
                if  PARAMS['clFunc']=='DNN-Ensemble':
                    Ensemble_Test_Params = DNN.test_dnn_ensemble_noise(PARAMS, Ensemble_Train_Params)
                elif PARAMS['clFunc']=='CNN-Ensemble':
                    Ensemble_Test_Params = CNN.test_cnn_ensemble_noise(PARAMS, Ensemble_Train_Params)

                kwargs = {}
                kwargs['0'] = 'feature_type:'+feature_type
                i = 1
                for dB in PARAMS['noise_dB_range']:
                    kwargs[str(i)] = str(dB)+'dB_ACC:'+str(Ensemble_Test_Params['accuracy_Ensemble'][dB])
                    kwargs[str(i+1)] = str(dB)+'dB_AVG_F1:'+str(Ensemble_Test_Params['fscore_Ensemble'][dB][2])
                    i += 2
                misc.print_results(PARAMS, 'noise_exp', **kwargs)
        
        if PARAMS['use_GPU']:
            reset_TF_session()
