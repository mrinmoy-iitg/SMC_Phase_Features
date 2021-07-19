#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:49:32 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
import configparser
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K
from sklearn.feature_extraction.image import PatchExtractor
import librosa
import lib.feature.preprocessing as preproc
from numba import jit, cuda




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



def get_feature_patches(PARAMS, Xin, segment_duration, segment_shift):
    Xin -= np.mean(Xin)
    Xin /= np.max(Xin)-np.min(Xin)
    # print('Xin: ', np.shape(Xin))
    seg_patches = np.empty([])

    if len(Xin)<segment_duration:
        Xin1 = Xin.copy()
        while len(Xin)<=segment_duration:
            Xin = np.append(Xin, Xin1)
    
    # startTime = time.process_time()
    numPatches = int(np.ceil(len(Xin)/segment_shift))
    Xin = np.expand_dims(np.expand_dims(Xin, axis=0), axis=2)
    # print('Xin: ', np.shape(Xin))
    seg_patches = PatchExtractor(patch_size=(segment_duration,1), max_patches=numPatches).transform(Xin)
    # print('sklearn splitting: ', time.process_time()-startTime, np.shape(seg_patches))

    return seg_patches


# @jit(target ="cuda")
def generator(PARAMS, folder, file_list, batchSize):
    batch_count = 0
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])

    file_list_temp = {0:file_list['music'].copy(), 1:file_list['speech'].copy()}
    batchData_temp = {0:np.empty([], dtype=float), 1:np.empty([], dtype=float)}
    balance = [0,0]
    while 1:
        batchData = np.empty([], dtype=float)
        batchLabel = np.empty([], dtype=float)

        for clNum in PARAMS['classes'].keys():
            while balance[clNum]<batchSize:
                if not file_list_temp[clNum]:
                    file_list_temp[clNum] = file_list[PARAMS['classes'][clNum]].copy()
                fName = file_list_temp[clNum].pop()
                fName_path = folder + '/' + PARAMS['classes'][clNum] + '/' + fName
                if not os.path.exists(fName_path):
                    continue    
                Xin, fs = librosa.core.load(fName_path, mono=True, sr=PARAMS['sampling_rate'])
                Xin = librosa.effects.preemphasis(Xin)
                Xin, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin, fs=fs, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], alpha=PARAMS['silThresh'], beta=0.1)
                segment_duration = int(PARAMS['CNN_patch_size']*PARAMS['Ts']*fs/1000)
                segment_shift = int(PARAMS['CNN_patch_shift']*PARAMS['Ts']*fs/1000)
                seg_patches = get_feature_patches(PARAMS, Xin, segment_duration, segment_shift)
                if balance[clNum]==0:
                    batchData_temp[clNum] = seg_patches
                else:
                    batchData_temp[clNum] = np.append(batchData_temp[clNum], seg_patches, axis=0)
                balance[clNum] += np.shape(seg_patches)[0]
        
        # print('balance: ', balance, np.shape(batchData_temp[0]), np.shape(batchData_temp[0]))
        batchData = batchData_temp[0][:batchSize, :]
        batchData = np.append(batchData, batchData_temp[1][:batchSize, :], axis=0)
        
        balance[0] -= batchSize
        balance[1] -= batchSize
        batchData_temp[0] = batchData_temp[0][batchSize:, :]
        batchData_temp[1] = batchData_temp[1][batchSize:, :]            

        class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        batchLabel = np.ones(2*batchSize)
        batchLabel[:batchSize] = class_labels['music']
        batchLabel[batchSize:] = class_labels['speech']
        OHE_batchLabel = to_categorical(batchLabel, num_classes=2)
                
        batch_count += 1
        # print('Batch ', batch_count, balance, ' shape=', np.shape(batchData), np.shape(OHE_batchLabel))
        yield batchData, OHE_batchLabel





def get_cnn_model(input_shape, num_classes):
    '''
    Baseline :- Doukhan et. al. MIREX 2018 MUSIC AND SPEECH DETECTION SYSTEM
    '''    
    input_img = Input(input_shape)
    
    x = Conv1D(39, kernel_size=400, strides=160, padding='same', kernel_regularizer=l2())(input_img)
    # print('feature lyr: ', K.int_shape(x))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = K.expand_dims(x, axis=3)
    # print('feature lyr: ', K.int_shape(x))
    
    x = Conv2D(64, kernel_size=(5, 4), strides=(1, 1), kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print('1: ', K.int_shape(x))

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # print('2: ', K.int_shape(x))
    
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print('3: ', K.int_shape(x))

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print('4: ', K.int_shape(x))

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # print('5: ', K.int_shape(x))
    
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print('6: ', K.int_shape(x))

    x = MaxPooling2D(pool_size=(12, 1), strides=(12, 1))(x)
    # print('7: ', K.int_shape(x))
    
    x = Flatten()(x)
    # print('8: ', K.int_shape(x))
    
    x = Dense(512, kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    # print('9: ', K.int_shape(x))

    x = Dense(512, kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    # print('10: ', K.int_shape(x))

    x = Dense(512, kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    # print('11: ', K.int_shape(x))

    x = Dense(512, kernel_regularizer=l2())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # print('12: ', K.int_shape(x))

    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2())(x)
    # print('13: ', K.int_shape(output))

    model = Model(input_img, output)
    learning_rate = 0.0001
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])    

    print('Architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n', model.summary())
    
    return model, learning_rate




def train_model(PARAMS, model, weightFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    logFile = '/'.join(weightFile.split('/')[:-2]) + '/log_fold' + str(PARAMS['fold']) + '.csv'
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.process_time()

    SPE = PARAMS['train_steps_per_epoch']
    SPE_val = PARAMS['val_steps']
    print('SPE: ', SPE, SPE_val)
    
    train_files = {}
    val_files = {}
    for classname in  PARAMS['train_files'].keys():
        files = PARAMS['train_files'][classname]
        np.random.shuffle(files)
        idx = int(len(files)*0.7)
        train_files[classname] = files[:idx]
        val_files[classname] = files[idx:]
    
    # Train the model
    History = model.fit(
            generator(PARAMS, PARAMS['folder'], train_files, PARAMS['batch_size']),
            steps_per_epoch = SPE,
            validation_data = generator(PARAMS, PARAMS['folder'], val_files, PARAMS['batch_size']), 
            validation_steps = SPE_val,
            epochs=PARAMS['epochs'], 
            verbose=1,
            callbacks=[csv_logger, es, mcp],
            # shuffle=True,
            )

    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History




def print_model_summary(arch_file, model):
    stringlist = ['Architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n']
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    with open(arch_file, 'w+', encoding='utf8') as f:
        f.write(short_model_summary)





def train_cnn(PARAMS):
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
        
    if not os.path.exists(paramFile):
        model, learning_rate = get_cnn_model(PARAMS['input_shape'], len(PARAMS['classes']))
        print_model_summary(arch_file, model)

        model, trainingTimeTaken, History = train_model(PARAMS, model, weightFile)
        
        if PARAMS['save_flag']:
            model.save_weights(weightFile) # Save the weights
            with open(architechtureFile, 'w') as f: # Save the model architecture
                f.write(model.to_json())
            np.savez(paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], input_shape=PARAMS['input_shape'], lr=learning_rate, trainingTimeTaken=trainingTimeTaken)
        print('CNN model trained.')
    else:
        PARAMS['epochs'] = np.load(paramFile)['epochs']
        PARAMS['batch_size'] = np.load(paramFile)['batch_size']
        PARAMS['input_shape'] = np.load(paramFile)['input_shape']
        learning_rate = np.load(paramFile)['lr']
        trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        optimizer = optimizers.Adam(lr=learning_rate)
        
        with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
            model = model_from_json(f.read())
        model.load_weights(weightFile) # Load weights into the new model

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print('CNN model exists! Loaded. Training time required=',trainingTimeTaken)
      
        
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': PARAMS['epochs'],
            'batch_size': PARAMS['batch_size'],
            'learning_rate': learning_rate,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def generator_test(PARAMS, file_name, clNum):
    fName_path = PARAMS['test_folder'] + '/' + PARAMS['classes'][clNum] + '/' + file_name
    Xin, fs = librosa.core.load(fName_path, mono=True, sr=PARAMS['sampling_rate'])
    Xin = librosa.effects.preemphasis(Xin)
    Xin, sample_silMarker, frame_silMarker, totalSilDuration = preproc.removeSilence(Xin=Xin, fs=fs, Tw=PARAMS['Tw'], Ts=PARAMS['Ts'], alpha=PARAMS['silThresh'], beta=0.1)
    segment_duration = int(PARAMS['CNN_patch_size']*PARAMS['Ts']*fs/1000)
    segment_shift = int(PARAMS['CNN_patch_shift']*PARAMS['Ts']*fs/1000)
    batchData = get_feature_patches(PARAMS, Xin, segment_duration, segment_shift)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel




def test_model_generator(PARAMS, Train_Params):
    loss = 0
    performance = 0
    testingTimeTaken = 0
    
    start = time.process_time()
    loss, performance = Train_Params['model'].evaluate_generator(
            generator(PARAMS, PARAMS['test_folder'], PARAMS['test_files'], PARAMS['batch_size']),
            steps=PARAMS['test_steps'], 
            verbose=1
            )
    print('loss: ', loss, ' performance: ', performance)
    testingTimeTaken = time.process_time() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    return loss, performance, testingTimeTaken


    
def test_model(PARAMS, Train_Params):
    start = time.process_time()
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    for classname in PARAMS['test_files'].keys():
        clNum = class_labels[classname]
        files = PARAMS['test_files'][classname]
        # print('test_files: ', files)
        fl_count = 0
        for fl in files:
            fl_count += 1
            fName = PARAMS['test_folder'] + '/' + classname + '/' + fl
            if not os.path.exists(fName):
                continue
            batchData, batchLabel = generator_test(PARAMS, fl, clNum)
            pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel.tolist())
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            print(fl_count, '/', len(files), PARAMS['classes'][clNum], fl, np.shape(batchData), ' acc=', np.round(np.sum(pred_lab==batchLabel)*100/len(batchLabel), 2))

    testingTimeTaken = time.process_time() - start
    print('Time taken for model testing: ',testingTimeTaken)
    ConfMat, fscore = misc.getPerformance(PtdLabels, GroundTruth)
    
    return ConfMat, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken




def test_cnn(PARAMS, Train_Params):
    loss = 0
    performance = 0
    loss, performance, testingTimeTaken = test_model_generator(PARAMS, Train_Params)
    
    ConfMat, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken = test_model(PARAMS, Train_Params)

    Test_Params = {
        'loss': loss,
        'performance': performance,
        'testingTimeTaken': testingTimeTaken,
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels_test': PtdLabels,
        'Predictions_test': Predictions,
        'GroundTruth_test': GroundTruth,
        }

    return Test_Params




def create_CV_folds(folder, opDir, classes, cv=3):
    if not os.path.exists(opDir+'/cv_file_list.pkl'):
        cv_file_list = {}
        for clNum in classes.keys():
            path = folder + '/' + classes[clNum] + '/'
            files = np.array(os.listdir(path))
            np.random.shuffle(files)
            files_per_fold = int(np.ceil(len(files)/cv))
            cv_file_list[classes[clNum]] = {}
            fl_count = 0
            for cv_num in range(cv):
                cv_file_list[classes[clNum]]['fold'+str(cv_num)] = files[fl_count:np.min([fl_count+files_per_fold, len(files)])]
                fl_count += files_per_fold
            
        misc.save_obj(cv_file_list, folder, 'cv_file_list')
        print('CV folds created')
    else:
        cv_file_list = misc.load_obj(folder, 'cv_file_list')
        print('\t\t\tCV folds loaded')
    return cv_file_list





def get_train_test_files(cv_file_list, cv_file_list_test, numCV, foldNum):
    train_files = {}
    test_files = {}
    for class_name in cv_file_list.keys():
        train_files[class_name] = []
        test_files[class_name] = []
        for i in range(numCV):
            files = cv_file_list[class_name]['fold'+str(i)]
            files_test = cv_file_list_test[class_name]['fold'+str(i)]
            if foldNum==i:
                test_files[class_name].extend(files_test)
            else:
                train_files[class_name].extend(files)
    
    return train_files, test_files






def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Raw_Audio_Classification.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': section['folder'],
            'output_folder': section['output_folder'],
            'test_folder': section['test_folder'],
            'dataset_size': float(section['dataset_size']), # GTZAN=1 / Scheirer-Slaney=1 / MUSNOMIX=3.6 / MUSAN=102
            'dataset_name': section['dataset_name'],
            'epochs': int(section['epochs']),
            'batch_size': int(section['batch_size']),
            'CV_folds': int(section['CV_folds']),
            'save_flag': section.getboolean('save_flag'),
            'data_generator': section.getboolean('data_generator'),
            'use_GPU': section.getboolean('use_GPU'),
            'scale_data': section.getboolean('scale_data'),
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'fold':0,
            'train_steps_per_epoch':0,
            'val_steps':0,
            'CNN_feat_dim': 0,
            'GPU_session':None,
            'opDir':'',
            'classes':{0:'music', 1:'speech'},
            'modelName':'',
            'sampling_rate': 16000,
            'Tw': 25,
            'Ts': 10,
            'silThresh': 0.025,
            }

    interval_shift = PARAMS['CNN_patch_shift']*10 # Frame shift in milisecs
    PARAMS['train_steps_per_epoch'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.7/(2*PARAMS['batch_size']))
    PARAMS['val_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.66*0.3/(2*PARAMS['batch_size']))
    PARAMS['test_steps'] = int(np.floor(PARAMS['dataset_size']*3600*1000/interval_shift)*0.33/(2*PARAMS['batch_size']))
    print('train_steps_per_epoch: %d, \tval_steps: %d,  \ttest_steps: %d\n'%(PARAMS['train_steps_per_epoch'], PARAMS['val_steps'], PARAMS['test_steps']))
        
    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()
    
    '''
    Initializations
    ''' 
    opDir_suffix = ''
    if PARAMS['test_folder']!='':
        opDir_suffix = '_GEN_PERF_' + PARAMS['folder'].split('/')[-3]
    PARAMS['opDir'] = PARAMS['output_folder'] + '/' + PARAMS['dataset_name'] + '/Raw_Audio_Classification/CNN' + opDir_suffix + '/'
    print('opDir: ', PARAMS['opDir'])
    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])

    if PARAMS['test_folder']=='':
        cv_file_list = create_CV_folds(PARAMS['folder'], PARAMS['opDir'], PARAMS['classes'], PARAMS['CV_folds'])
        cv_file_list_test = cv_file_list
        PARAMS['test_folder'] = PARAMS['folder']
        
    else:
        cv_file_list = create_CV_folds(PARAMS['folder'], PARAMS['opDir'], PARAMS['classes'], PARAMS['CV_folds'])
        cv_file_list_test = misc.create_CV_folds(PARAMS['test_folder'], PARAMS['opDir'], PARAMS['classes'], PARAMS['CV_folds'])
    
    misc.print_configuration(PARAMS)
                    
    for foldNum in range(PARAMS['CV_folds']): # range(1)
        PARAMS['fold'] = foldNum
        PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)
        
        PARAMS['modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        PARAMS['input_shape'] = (int(PARAMS['CNN_patch_size']*PARAMS['Ts']*PARAMS['sampling_rate']/1000), 1)

        if PARAMS['use_GPU']:
            start_GPU_session()

        Train_Params = train_cnn(PARAMS)

        Test_Params = test_cnn(PARAMS, Train_Params)
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

        Train_Params = None
        Test_Params = None
        if PARAMS['use_GPU']:
            reset_TF_session()        


