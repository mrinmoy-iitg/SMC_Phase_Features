#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:56:47 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
import configparser
import lib.classifier.multi_class_cnn_classifier as MCNN



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
    section = config['Multi_Class_Classification.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': section['folder'],
            'clFunc': 'CNN',
            'dataset_size': float(section['dataset_size']), # MUSNOMIX=2.6 / MUSAN=102
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
            'classes':{0:'music', 1:'speech', 2:'music+noise', 3:'noise', 4:'speech+music', 5:'speech+music+noise', 6:'speech+noise'},
            'all_featName': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'],
            'featName':'',
            'modelName':'',
            }

    interval_shift = PARAMS['CNN_patch_shift']*10 # Frame shift in milisecs
    PARAMS['train_steps_per_epoch'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.7/(len(PARAMS['classes'])*PARAMS['batch_size']))
    PARAMS['val_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.3/(len(PARAMS['classes'])*PARAMS['batch_size']))
    PARAMS['test_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.33/(len(PARAMS['classes'])*PARAMS['batch_size']))
    print('train_steps_per_epoch: %d, \tval_steps: %d,  \ttest_steps: %d\n'%(PARAMS['train_steps_per_epoch'], PARAMS['val_steps'], PARAMS['test_steps']))
    
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    
    for PARAMS['featName'] in PARAMS['all_featName']:
        PARAMS['input_shape'] = (21, PARAMS['CNN_patch_size'], 1)
        print(PARAMS['featName'])

        '''
        Initializations
        ''' 
        cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
        cv_file_list_test = cv_file_list
        PARAMS['test_folder'] = PARAMS['folder']
        PARAMS['output_folder'] = PARAMS['test_folder'] + '/' + PARAMS['featName'] + '/__RESULTS/' + PARAMS['today'] + '/'
                        
        if not os.path.exists(PARAMS['output_folder']):
            os.makedirs(PARAMS['output_folder'])
        
        PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['clFunc'] + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
        
        misc.print_configuration(PARAMS)
                    
        for foldNum in range(PARAMS['CV_folds']):
            PARAMS['fold'] = foldNum
            PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
            
            PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
            if PARAMS['use_GPU']:
                start_GPU_session()

            Train_Params = MCNN.train_cnn(PARAMS)
            
            Test_Params = MCNN.test_cnn(PARAMS, Train_Params)
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
                    '8':'F_score_muno:'+str(Test_Params['fscore'][2]),
                    '9':'F_score_no:'+str(Test_Params['fscore'][3]),
                    '10':'F_score_spmu:'+str(Test_Params['fscore'][4]),
                    '11':'F_score_spmuno:'+str(Test_Params['fscore'][5]),
                    '12':'F_score_spno:'+str(Test_Params['fscore'][6]),
                    '13':'F_score_avg:'+str(Test_Params['fscore'][7]),
                    }
            misc.print_results(PARAMS, '', **kwargs)

            Train_Params = None
            Test_Params = None
            if PARAMS['use_GPU']:
                reset_TF_session()        
