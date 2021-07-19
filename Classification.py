#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:51:23 2018
Updated on Tue Apr  13 17:15:23 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
import configparser
import sys



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




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Classification.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': section['folder'],
            'test_path': section['test_path'],
            'dataset_size': float(section['dataset_size']), # GTZAN=1 / Scheirer-Slaney=1 / MUSNOMIX=3.6 / MUSAN=102
            'dataset_name': section['dataset_name'],
            'clFunc': section['clFunc'], # DNN-GridSearch / DNN / CNN / SVM / NB
            'experiment_type': section['experiment_type'], # training_testing / testing
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'epochs': int(section['epochs']),
            'batch_size': int(section['batch_size']),
            'CV_folds': int(section['CV_folds']),
            'save_flag': section.getboolean('save_flag'),
            'data_generator': section.getboolean('data_generator'),
            'data_balancing': section.getboolean('data_balancing'),
            'use_GPU': section.getboolean('use_GPU'),
            'scale_data': section.getboolean('scale_data'),
            'PCA_flag': section.getboolean('PCA_flag'),
            'fold':0,
            'train_steps_per_epoch':0,
            'val_steps':0,
            'CNN_feat_dim': 0,
            'GPU_session':None,
            'output_folder':'',
            'test_folder': '',
            'opDir':'',
            'classes':{0:'music', 1:'speech'},
            'all_featName': ['MFCC-39'], 
            # 'all_featName': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'], # 'MGDCC-39_row_gamma'
            '39_dim_CC_feat': section.getboolean('39_dim_CC_feat'),
            'featName':'',
            'num_dnn_lyr': [2, 3, 4, 5],
            'num_dnn_nodes': [100, 250, 500, 1000],
            'DNN_optimal_params': {
                'MUSAN':
                {'Khonglah_et_al':{'hidden_lyrs':3, 'hidden_nodes':250},
                'Sell_et_al':{'hidden_lyrs':2, 'hidden_nodes':100},
                'MFCC-39':{'hidden_lyrs':3, 'hidden_nodes':100}, #updated
                'Melspectrogram':{'hidden_lyrs':5, 'hidden_nodes':250},
                'HNGDMFCC':{'hidden_lyrs':5, 'hidden_nodes':100}, #updated
                'MGDCC':{'hidden_lyrs':3, 'hidden_nodes':100}, #updated
                'IFCC':{'hidden_lyrs':4, 'hidden_nodes':250},}, #updated
                'MUSNOMIX_WAV':
                {'Khonglah_et_al':{'hidden_lyrs':5, 'hidden_nodes':250}, #updated
                'Sell_et_al':{'hidden_lyrs':3, 'hidden_nodes':1000}, #updated
                'MFCC-39':{'hidden_lyrs':4, 'hidden_nodes':500}, #updated
                'Melspectrogram':{'hidden_lyrs':2, 'hidden_nodes':1000}, #updated
                'HNGDMFCC':{'hidden_lyrs':3, 'hidden_nodes':500}, #updated
                'MGDCC':{'hidden_lyrs':2, 'hidden_nodes':1000}, #updated
                'IFCC':{'hidden_lyrs':5, 'hidden_nodes':500}} #updated
                },
            'modelName':'',
            'noise_experiment': section.getboolean('noise_experiment'),
            'noise_dB_range': [10, 8, 5, 2, 1, 0],
            }

    interval_shift = PARAMS['CNN_patch_shift']*10 # Frame shift in milisecs
    PARAMS['train_steps_per_epoch'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.7/(2*PARAMS['batch_size']))
    PARAMS['val_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.3/(2*PARAMS['batch_size']))
    PARAMS['test_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.33/(2*PARAMS['batch_size']))
    print('train_steps_per_epoch: %d, \tval_steps: %d,  \ttest_steps: %d\n'%(PARAMS['train_steps_per_epoch'], PARAMS['val_steps'], PARAMS['test_steps']))
        
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    
    for PARAMS['featName'] in PARAMS['all_featName']:
        
        if PARAMS['featName'] == 'MGDCC-39_row_gamma':
            if len(sys.argv)<2:
                sys.exit(0)
            PARAMS['rho'] = np.round(float(sys.argv[1]),1)
            PARAMS['gamma'] = np.round(float(sys.argv[2]),1)
            PARAMS['featName'] = 'MGDCC-39_rho' + str(PARAMS['rho']) + '_gamma' + str(PARAMS['gamma'])
            if not os.path.exists(PARAMS['folder']+'/'+PARAMS['featName']+'/'):
                print(PARAMS['featName'], ' does not exist')
                sys.exit(0)
            print(PARAMS['featName'])

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
        PARAMS['input_shape'] = (21, PARAMS['CNN_patch_size'], 1)
        
        if PARAMS['39_dim_CC_feat']:
            if (PARAMS['featName']=='HNGDMFCC') or (PARAMS['featName']=='MGDCC') or (PARAMS['featName']=='IFCC') or (PARAMS['featName']=='MFCC-39'):
                PARAMS['CNN_feat_dim'] = 39
                PARAMS['input_dim'] = 78
                PARAMS['input_shape'] = (39, PARAMS['CNN_patch_size'], 1)
            
        print(PARAMS['featName'], PARAMS['input_shape'], PARAMS['CNN_feat_dim'])

        '''
        Initializations
        ''' 
        opDir_suffix = ''
        if PARAMS['test_path']=='':
            cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
            cv_file_list_test = cv_file_list
            PARAMS['test_folder'] = PARAMS['folder']
            PARAMS['output_folder'] = PARAMS['test_folder'] + '/' + PARAMS['featName'] + '/__RESULTS/' + PARAMS['today'] + '/'
            
        else:
            PARAMS['test_folder'] = PARAMS['test_path']
            cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
            cv_file_list_test = misc.create_CV_folds(PARAMS['test_folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
            PARAMS['output_folder'] = PARAMS['test_folder'] + '/' + PARAMS['featName'] + '/__RESULTS/' + PARAMS['today'] + '/'
            if PARAMS['noise_experiment']:
                opDir_suffix = '_Noise_Experiment'
            else:
                opDir_suffix = '_GEN_PERF_' + PARAMS['folder'].split('/')[-3]
            
        if not os.path.exists(PARAMS['output_folder']):
            os.makedirs(PARAMS['output_folder'])
        
        if PARAMS['39_dim_CC_feat']:
            if (PARAMS['featName']=='HNGDMFCC') or (PARAMS['featName']=='MGDCC') or (PARAMS['featName']=='IFCC') or (PARAMS['featName']=='MFCC-39'):
                opDir_suffix += '_39CC'
        PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['clFunc'] + opDir_suffix + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
        
        misc.print_configuration(PARAMS)
        
        All_Folds_Results = np.empty([])
                    
        for foldNum in range(PARAMS['CV_folds']):
            PARAMS['fold'] = foldNum
            PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
            
            '''
            Load data
            '''
            if not PARAMS['data_generator']:
                train_data, train_label = misc.load_data_from_files(PARAMS['classes'], PARAMS['folder'], PARAMS['featName'], PARAMS['train_files'], PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
                test_data, test_label = misc.load_data_from_files(PARAMS['classes'], PARAMS['test_folder'], PARAMS['featName'], PARAMS['test_files'], PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])
                print('Data loaded: ', np.shape(train_data), np.shape(train_label), np.shape(test_data), np.shape(test_label))
                
                train_data, train_label, test_data = misc.preprocess_data(PARAMS, train_data, train_label, test_data)
                print('Data preprocessed: ', np.shape(train_data), np.shape(test_data))
            else:
                train_data = np.empty([])
                train_label = np.empty([])
                test_data = np.empty([])
                test_label = np.empty([])
                

            if PARAMS['clFunc']=='DNN-GridSearch':
                import lib.classifier.dnn_classifier as DNN
                for num_lyr in PARAMS['num_dnn_lyr']:
                    for num_nodes in PARAMS['num_dnn_nodes']:
                        print('DNN classification for num_lyr=', num_lyr, ' num_nodes=', num_nodes)
                        PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_lyrs' + str(num_lyr) + '_nodes' + str(num_nodes) + '_model.xyz'
                        if PARAMS['use_GPU']:
                            start_GPU_session()

                        Train_Params = DNN.train_dnn(PARAMS, train_data, train_label, num_lyr, num_nodes)
                        Test_Params = DNN.test_dnn(PARAMS, test_data, test_label, Train_Params)
            
                        print('Test accuracy=', Test_Params['fscore'])
            
                        kwargs = {
                                '0':'epochs:'+str(Train_Params['epochs']),
                                '1':'batch_size:'+str(Train_Params['batch_size']),
                                '2':'learning_rate:'+str(Train_Params['learning_rate']),
                                '3':'training_time:'+str(Train_Params['trainingTimeTaken']),
                                '4':'Hidden Layers:'+str(num_lyr),
                                '5':'Hidden Nodes:'+str(num_nodes),
                                '6':'loss:'+str(Test_Params['loss']),
                                '7':'performance:'+str(Test_Params['performance']),
                                '8':'F_score_mu:'+str(Test_Params['fscore'][0]),
                                '9':'F_score_sp:'+str(Test_Params['fscore'][1]),
                                '10':'F_score_avg:'+str(Test_Params['fscore'][2]),
                                }
                        misc.print_results(PARAMS, '', **kwargs)
                        Train_Params = None
                        Test_Params = None
                        if PARAMS['use_GPU']:
                            reset_TF_session()        



            elif PARAMS['clFunc']=='DNN':
                import lib.classifier.dnn_classifier as DNN
                if PARAMS['experiment_type']=='training_testing':
                    PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                elif PARAMS['experiment_type']=='testing':
                    PARAMS['modelName'] = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/DNN/fold' + str(PARAMS['fold']) + '_model.xyz'
                    
                if PARAMS['use_GPU']:
                    start_GPU_session()
                    
                num_lyr = PARAMS['DNN_optimal_params'][PARAMS['dataset_name']][PARAMS['featName']]['hidden_lyrs']
                num_nodes = PARAMS['DNN_optimal_params'][PARAMS['dataset_name']][PARAMS['featName']]['hidden_nodes']

                Train_Params = DNN.train_dnn(PARAMS, train_data, train_label, num_lyr, num_nodes)

                if not PARAMS['noise_experiment']:
                    Test_Params = DNN.test_dnn(PARAMS, test_data, test_label, Train_Params)    
                    print('Test accuracy=', Test_Params['fscore'])    
                    kwargs = {
                            '0':'epochs:'+str(Train_Params['epochs']),
                            '1':'batch_size:'+str(Train_Params['batch_size']),
                            '2':'learning_rate:'+str(Train_Params['learning_rate']),
                            '3':'training_time:'+str(Train_Params['trainingTimeTaken']),
                            '4':'Hidden Layers:'+str(num_lyr),
                            '5':'Hidden Nodes:'+str(num_nodes),
                            '6':'loss:'+str(Test_Params['loss']),
                            '7':'performance:'+str(Test_Params['performance']),
                            '8':'F_score_mu:'+str(Test_Params['fscore'][0]),
                            '9':'F_score_sp:'+str(Test_Params['fscore'][1]),
                            '10':'F_score_avg:'+str(Test_Params['fscore'][2]),
                            }
                    misc.print_results(PARAMS, '', **kwargs)
                else:
                    Test_Params = DNN.test_dnn_noise(PARAMS, Train_Params)
                    kwargs = {}
                    kwargs['0'] = 'featName:'+PARAMS['featName']
                    i = 1
                    for dB in PARAMS['noise_dB_range']:
                        kwargs[str(i)] = str(dB)+'dB_ACC:'+str(Test_Params['accuracy'][dB])
                        kwargs[str(i+1)] = str(dB)+'dB_AVG_F1:'+str(Test_Params['fscore'][dB][2])
                        i += 2
                    misc.print_results(PARAMS, 'noise_exp', **kwargs)

                Train_Params = None
                Test_Params = None
                if PARAMS['use_GPU']:
                    reset_TF_session()        



            elif PARAMS['clFunc']=='CNN':
                import lib.classifier.cnn_classifier as CNN
                if PARAMS['experiment_type']=='training_testing':
                    PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                elif PARAMS['experiment_type']=='testing':
                    PARAMS['modelName'] = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/CNN/fold' + str(PARAMS['fold']) + '_model.xyz'

                if PARAMS['use_GPU']:
                    start_GPU_session()

                Train_Params = CNN.train_cnn(PARAMS)
                
                if not PARAMS['noise_experiment']:
                    Test_Params = CNN.test_cnn(PARAMS, Train_Params)
                    print('Test accuracy=', Test_Params['fscore'])
                    kwargs = {
                            '0':'epochs:'+str(Train_Params['epochs']),
                            '1':'batch_size:'+str(Train_Params['batch_size']),
                            '2':'learning_rate:'+str(Train_Params['learning_rate']),
                            '3':'training_time:'+str(Train_Params['trainingTimeTaken']),
                            '4':'loss:'+str(Test_Params['loss']),
                            '5':'performance:'+str(Test_Params['performance']),
                            '6':'F_score_mu:'+str(Test_Params['fscore'][0]),
                            '7':'F_score_sp:'+str(Test_Params['fscore'][1]),
                            '8':'F_score_avg:'+str(Test_Params['fscore'][2]),
                            }
                    misc.print_results(PARAMS, '', **kwargs)
                else:
                    Test_Params = CNN.test_cnn_noise(PARAMS, Train_Params)
                    kwargs = {}
                    kwargs['0'] = 'featName:'+PARAMS['featName']
                    i = 1
                    for dB in PARAMS['noise_dB_range']:
                        kwargs[str(i)] = str(dB)+'dB_ACC:'+str(Test_Params['accuracy'][dB])
                        kwargs[str(i+1)] = str(dB)+'dB_AVG_F1:'+str(Test_Params['fscore'][dB][2])
                        i += 2
                    misc.print_results(PARAMS, 'noise_exp', **kwargs)
    
                Train_Params = None
                Test_Params = None
                if PARAMS['use_GPU']:
                    reset_TF_session()        


            elif PARAMS['clFunc']=='SVM':
                import lib.classifier.svm_classifier as SVM
                if PARAMS['experiment_type']=='training_testing':
                    PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                elif PARAMS['experiment_type']=='testing':
                    PARAMS['modelName'] = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/SVM/fold' + str(PARAMS['fold']) + '_model.xyz'
                    
                Train_Params, Test_Params = SVM.grid_search_svm(PARAMS, train_data, train_label, test_data, test_label)
                print('Test accuracy=', Test_Params['fscore'])    
                kwargs = {
                        '0':'training_time:'+str(Train_Params['trainingTimeTaken']),
                        '1':'cost:'+str(Train_Params['optC']),
                        '2':'gamma:'+str(Train_Params['optGamma']),
                        '3':'accuracy:'+str(Test_Params['accuracy']),
                        '4':'F_score_mu:'+str(Test_Params['fscore'][0]),
                        '5':'F_score_sp:'+str(Test_Params['fscore'][1]),
                        '6':'F_score_avg:'+str(Test_Params['fscore'][2]),
                        }
                misc.print_results(PARAMS, '', **kwargs)
                Train_Params = None
                Test_Params = None



            elif PARAMS['clFunc']=='NB':
                import lib.classifier.NB_classifier as NB
                if PARAMS['experiment_type']=='training_testing':
                    PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
                elif PARAMS['experiment_type']=='testing':
                    PARAMS['modelName'] = PARAMS['folder'] + '/' + PARAMS['featName'] + '/__RESULTS/NB/fold' + str(PARAMS['fold']) + '_model.xyz'
                    
                Train_Params, Test_Params = NB.naive_bayes_classification(PARAMS, train_data, train_label, test_data, test_label)
                print('Test accuracy=', Test_Params['fscore'])    
                kwargs = {
                        '0':'training_time:'+str(Train_Params['trainingTimeTaken']),
                        '1':'accuracy:'+str(Test_Params['accuracy']),
                        '2':'F_score_mu:'+str(Test_Params['fscore'][0]),
                        '3':'F_score_sp:'+str(Test_Params['fscore'][1]),
                        '4':'F_score_avg:'+str(Test_Params['fscore'][2]),
                        }
                misc.print_results(PARAMS, '', **kwargs)
                
                if PARAMS['featName'].startswith('MGDCC-39_rho'):
                    if np.size(All_Folds_Results)<=1:
                        All_Folds_Results = np.array([Test_Params['accuracy'], Test_Params['fscore'][0], Test_Params['fscore'][1], Test_Params['fscore'][2]], ndmin=2)
                    else:
                        All_Folds_Results = np.append(All_Folds_Results, np.array([Test_Params['accuracy'], Test_Params['fscore'][0], Test_Params['fscore'][1], Test_Params['fscore'][2]], ndmin=2), axis=0)

                Train_Params = None
                Test_Params = None
                    
        
        
        '''
        Only for MGDCC_rho_gamma
        '''
        if PARAMS['featName'].startswith('MGDCC-39_rho'):
            PARAMS['opDir'] = PARAMS['folder'] + '/__RESULTS/' + PARAMS['today'] + '/MGDCC_rho_gamma/'
            if not os.path.exists(PARAMS['opDir']):
                os.makedirs(PARAMS['opDir'])
            print(np.mean(All_Folds_Results, axis=0))
            kwargs = {
                    '0':'rho:'+str(PARAMS['rho']),
                    '1':'gamma:'+str(PARAMS['gamma']),
                    '2':'accuracy:'+str(np.mean(All_Folds_Results[:,0])),
                    '3':'accuracy:'+str(np.mean(All_Folds_Results[:,0])),
                    '4':'F_score_mu:'+str(np.mean(All_Folds_Results[:,1])),
                    '5':'F_score_sp:'+str(np.mean(All_Folds_Results[:,2])),
                    '6':'F_score_avg:'+str(np.mean(All_Folds_Results[:,3])),
                    '7':'F_score_stdev:'+str(np.std(All_Folds_Results[:,3])),
                    }
            misc.print_results(PARAMS, '', **kwargs)
            
