#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:00:47 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input
import lib.misc as misc
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import PatchExtractor
import scipy




def get_feature_patches(PARAMS, FV, patch_size, patch_shift, input_shape):
    # Removing NaN and Inf
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV[~np.isnan(FV).any(axis=1), :]
        FV = FV[~np.isinf(FV).any(axis=1), :]
    else:
        FV = FV[:, ~np.isnan(FV).any(axis=0)]
        FV = FV[:, ~np.isinf(FV).any(axis=0)]

    FV = StandardScaler(copy=False).fit_transform(FV)
    # FV should be of the shape (nFeatures, nFrames)
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV.T
                
    frmStart = 0
    frmEnd = 0
    patchNum = 0
    patches = np.empty([])

    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)

    # while frmEnd<np.shape(FV)[1]:
    #     # print('get_feature_patches: ', frmStart, frmEnd, np.shape(FV))
    #     frmStart = patchNum*patch_shift
    #     frmEnd = np.min([patchNum*patch_shift+patch_size, np.shape(FV)[1]])
    #     if frmEnd-frmStart<patch_size:
    #         frmStart = frmEnd - patch_size
    #     if np.size(patches)<=1:
    #         patches = np.expand_dims(FV[:, frmStart:frmEnd], axis=0)
    #     else:
    #         patches = np.append(patches, np.expand_dims(FV[:, frmStart:frmEnd], axis=0), axis=0)
    #     patchNum += 1


    # startTime = time.clock()
    # for frmStart in range(0, np.shape(FV)[1], patch_shift):
    #     # print('get_feature_patches: ', frmStart, frmEnd, np.shape(FV))
    #     frmEnd = np.min([frmStart+patch_size, np.shape(FV)[1]])
    #     if frmEnd-frmStart<patch_size:
    #         frmStart = frmEnd - patch_size
    #     if np.size(patches)<=1:
    #         patches = np.array(FV[:, frmStart:frmEnd], ndmin=3)
    #     else:
    #         patches = np.append(patches, np.array(FV[:, frmStart:frmEnd], ndmin=3), axis=0)
    # print('My splitting: ', time.clock()-startTime, np.shape(patches))
    
    
    startTime = time.clock()
    numPatches = int(np.ceil(np.shape(FV)[1]/patch_shift))
    patches = PatchExtractor(patch_size=(np.shape(FV)[0], patch_size), max_patches=numPatches).transform(np.expand_dims(FV, axis=0))
    # print('sklearn splitting: ', time.clock()-startTime, np.shape(patches))


    # print('Patches: ', np.shape(patches))
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



def generator(PARAMS, folder, file_list, batchSize):
    batch_count = 0
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])

    file_list_sp_temp = file_list['speech'].copy()
    file_list_mu_temp = file_list['music'].copy()

    batchData_sp = np.empty([], dtype=float)
    batchData_mu = np.empty([], dtype=float)
    balance_sp = 0
    balance_mu = 0
    while 1:
        batchData = np.empty([], dtype=float)
        batchLabel = np.empty([], dtype=float)

        while balance_sp<batchSize:
            if not file_list_sp_temp:
                file_list_sp_temp = file_list['speech'].copy()
            sp_fName = file_list_sp_temp.pop()
            sp_fName_path = folder + '/' + PARAMS['featName'] + '/speech/' + sp_fName
            if not os.path.exists(sp_fName_path):
                continue
            # print('sp_fName_path: ', sp_fName_path)
            fv_sp = np.load(sp_fName_path, allow_pickle=True)
            fv_sp = get_feature_patches(PARAMS, fv_sp, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
            fv_sp = np.expand_dims(fv_sp, axis=3)
            if balance_sp==0:
                batchData_sp = fv_sp
            else:
                batchData_sp = np.append(batchData_sp, fv_sp, axis=0)
            balance_sp += np.shape(fv_sp)[0]
            # print('Speech: ', batchSize, balance_sp, np.shape(batchData_sp))
            
        while balance_mu<batchSize:
            if not file_list_mu_temp:
                file_list_mu_temp = file_list['music'].copy()
            mu_fName = file_list_mu_temp.pop()
            mu_fName_path = folder + '/' + PARAMS['featName'] + '/music/' + mu_fName
            if not os.path.exists(mu_fName_path):
                continue
            # print('mu_fName_path: ', mu_fName_path)
            fv_mu = np.load(mu_fName_path, allow_pickle=True)
            fv_mu = get_feature_patches(PARAMS, fv_mu, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
            fv_mu = np.expand_dims(fv_mu, axis=3)
            if balance_mu==0:
                batchData_mu = fv_mu
            else:
                batchData_mu = np.append(batchData_mu, fv_mu, axis=0)
            balance_mu += np.shape(fv_mu)[0]
            # print('Music: ', batchSize, balance_mu, np.shape(batchData_mu))
                
        batchData = batchData_sp[:batchSize, :]
        batchData = np.append(batchData, batchData_mu[:batchSize, :], axis=0)
        
        balance_sp -= batchSize
        balance_mu -= batchSize
        batchData_sp = batchData_sp[batchSize:, :]
        batchData_mu = batchData_mu[batchSize:, :]            

        class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        batchLabel = np.ones(2*batchSize)
        batchLabel[:batchSize] = class_labels['speech']
        batchLabel[batchSize:] = class_labels['music']
        OHE_batchLabel = to_categorical(batchLabel, num_classes=2)
                
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData))
        yield batchData, OHE_batchLabel




def train_model(PARAMS, model, weightFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    logFile = '/'.join(weightFile.split('/')[:-2]) + '/log_fold' + str(PARAMS['fold']) + '.csv'
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.clock()

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
    History = model.fit_generator(
            generator(PARAMS, PARAMS['folder'], train_files, PARAMS['batch_size']),
            steps_per_epoch = SPE,
            validation_data = generator(PARAMS, PARAMS['folder'], val_files, PARAMS['batch_size']), 
            validation_steps = SPE_val,
            epochs=PARAMS['epochs'], 
            verbose=1,
            callbacks=[csv_logger, es, mcp],
            # shuffle=True,
            )

    trainingTimeTaken = time.clock() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History




def print_model_summary(arch_file, model):
    stringlist = ['Architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n']
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    with open(arch_file, 'w+', encoding='utf8') as f:
        f.write(short_model_summary)




def get_cnn_model(input_shape, num_classes):
    '''
    Baseline :- Doukhan et. al. MIREX 2018 MUSIC AND SPEECH DETECTION SYSTEM
    '''
    # import tensorflow.keras.backend as K
    
    input_img = Input((input_shape[0], input_shape[1], input_shape[2]))
    # print('0: ', K.int_shape(input_img))

    x = Conv2D(64, kernel_size=(4, 5), strides=(1, 1), kernel_regularizer=l2())(input_img)
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

    x = MaxPooling2D(pool_size=(1, 12), strides=(1, 12))(x)
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
        
        # with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
        #     model = model_from_json(f.read())
        try:
            with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
                model = model_from_json(f.read())
        except:
            model, learning_rate_temp = get_cnn_model(PARAMS['input_shape'], 2)
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
        



def generator_test(PARAMS, featName, file_name, clNum):
    fName = PARAMS['test_folder'] + '/' + featName + '/'+ PARAMS['classes'][clNum] + '/' + file_name
    # startTime = time.clock()
    batchData = np.load(fName, allow_pickle=True)
    # print('Loading: ', time.clock()-startTime)

    # startTime = time.clock()
    batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])
    # print('Splitting: ', time.clock()-startTime)

    batchData = np.expand_dims(batchData, axis=3)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel




def test_model_generator(PARAMS, Train_Params):
    loss = 0
    performance = 0
    testingTimeTaken = 0
    
    start = time.clock()
    loss, performance = Train_Params['model'].evaluate_generator(
            generator(PARAMS, PARAMS['test_folder'], PARAMS['test_files'], PARAMS['batch_size']),
            steps=PARAMS['test_steps'], 
            verbose=1
            )
    print('loss: ', loss, ' performance: ', performance)
    testingTimeTaken = time.clock() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    return loss, performance, testingTimeTaken


    
def test_model(PARAMS, Train_Params):
    start = time.clock()
    PtdLabels = []
    GroundTruth = []
    Predictions = np.empty([])
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    startTime = time.clock()
    for classname in PARAMS['test_files'].keys():
        clNum = class_labels[classname]
        files = PARAMS['test_files'][classname]
        # print('test_files: ', files)
        for fl in files:
            fName = PARAMS['test_folder'] + '/' + PARAMS['featName'] + '/'+ classname + '/' + fl
            if not os.path.exists(fName):
                continue
            count += 1
            batchData, batchLabel = generator_test(PARAMS, PARAMS['featName'], fl, clNum)
            endTime = time.clock()
            print('Data loading time: ', endTime-startTime)
            
            startTime = time.clock()
            pred = Train_Params['model'].predict(x=batchData)
            print('Prediction time: ', time.clock()-startTime)

            pred_lab = np.argmax(pred, axis=1)
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel.tolist())
            print('pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1))
            print('ground_truth: ', np.sum(batchLabel==0), np.sum(batchLabel==1))
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            print(PARAMS['classes'][clNum], fl, np.shape(batchData), ' acc=', np.round(np.sum(pred_lab==batchLabel)*100/len(batchLabel), 2))

    testingTimeTaken = time.clock() - start
    print('Time taken for model testing: ',testingTimeTaken)
    ConfMat, fscore = misc.getPerformance(PtdLabels, GroundTruth)
    
    return ConfMat, fscore, PtdLabels, Predictions, GroundTruth, testingTimeTaken




def test_cnn(PARAMS, Train_Params):
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




def generator_test_ensemble(PARAMS, featName, file_name, clNum):
    fName = PARAMS['test_folder'] + '/' + featName + '/'+ PARAMS['classes'][clNum] + '/' + file_name
    # startTime = time.clock()
    batchData = np.load(fName, allow_pickle=True)
    # print('Loading: ', time.clock()-startTime)

    # startTime = time.clock()
    # print(np.shape(batchData))
    # print(PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])
    batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'][featName])
    # print('Splitting: ', time.clock()-startTime)

    batchData = np.expand_dims(batchData, axis=3)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel




def test_cnn_ensemble(PARAMS, Ensemble_Train_Params):
    start = time.clock()
    PtdLabels_Ensemble = []
    GroundTruth_Ensemble = []
    Predictions_Ensemble = np.empty([])
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    individual_performances = {
        'Khonglah_et_al':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]}, 
        'Sell_et_al':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]}, 
        'MFCC-39':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]}, 
        'Melspectrogram':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]}, 
        'HNGDMFCC':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]},
        'MGDCC':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]}, 
        'IFCC':{'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]},
        }
    temp_folder = PARAMS['opDir'] + '/__temp/'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    for classname in PARAMS['test_files'].keys():
        clNum = class_labels[classname]
        files = PARAMS['test_files'][classname]
        # print('test_files: ', files)
        fl_count = 0
        for fl in files:
            fl_count += 1
            print('\t\t\t',PARAMS['fold'], PARAMS['classes'][clNum], fl, fl_count, '/', len(files), end='\t')
            count += 1
            PtdLabels = None
            PtdLabels_temp = np.empty([])
            GroundTruth = np.empty([])
            Predictions = np.empty([])
            empty_predictions = False
            for featName in Ensemble_Train_Params.keys():

                empty_predictions = False
                curr_fName = PARAMS['test_folder'] + '/' + featName + '/'+ classname + '/' + fl
                # print('curr_fName: ', curr_fName)
                if not os.path.exists(curr_fName):
                    # print('curr_file not found')
                    empty_predictions = True
                    break

                Train_Params = Ensemble_Train_Params[featName]
                batchData = None
                batchLabel = None
                temp_file = 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + fl.split('.')[0]
                # print(temp_folder, temp_file)
                # print(featName, Train_Params['model'].layers[0].output_shape, PARAMS['input_shape'][featName])
                if not os.path.exists(temp_folder + '/' + temp_file + '.pkl'):
                    batchData, batchLabel = generator_test_ensemble(PARAMS, featName, fl, clNum)
                    # print('batchData: ', np.shape(batchData), np.shape(batchLabel))
                    pred = Train_Params['model'].predict(x=batchData)
                    # print('pred: ', np.shape(pred))
                    misc.save_obj(pred, temp_folder, temp_file)
                else:
                    try:
                        pred = misc.load_obj(temp_folder, temp_file)
                    except:
                        batchData, batchLabel = generator_test_ensemble(PARAMS, featName, fl, clNum)
                        # print('batchData: ', np.shape(batchData), np.shape(batchLabel))
                        pred = Train_Params['model'].predict(x=batchData)
                        # print('pred: ', np.shape(pred))
                        misc.save_obj(pred, temp_folder, temp_file)
                        
                # print('indv_labels: ', np.shape(indv_labels), np.shape(individual_performances[featName]['PtdLabels']))
                if np.size(individual_performances[featName]['Predictions'])<=1:
                    individual_performances[featName]['Predictions'] = np.array(pred, ndmin=2)
                    individual_performances[featName]['GroundTruth'] = np.ones(np.shape(pred)[0])*clNum
                else:
                    individual_performances[featName]['Predictions'] = np.append(individual_performances[featName]['Predictions'], np.array(pred, ndmin=2), axis=0)
                    individual_performances[featName]['GroundTruth'] = np.append(individual_performances[featName]['GroundTruth'], np.ones(np.shape(pred)[0])*clNum)

                if np.size(Predictions)<=1:
                    Predictions = np.array(pred, ndmin=2)
                    PtdLabels_temp = np.array(np.argmax(pred, axis=1), ndmin=2).T
                else:
                    # print('PtdLabels_temp: ', np.shape(PtdLabels_temp), np.shape(pred))
                    empty_predictions = False
                    if np.shape(pred)[0]!=np.shape(Predictions)[0]:
                        if np.shape(pred)[0]>np.shape(Predictions)[0]:
                            pred = pred[:np.shape(Predictions)[0], :]
                        else:
                            empty_predictions = True
                            break
                    Predictions = np.add(Predictions, np.array(pred, ndmin=2))
                    PtdLabels_temp = np.append(PtdLabels_temp, np.array(np.argmax(pred, axis=1), ndmin=2).T, axis=1)
            
            if empty_predictions:
                print(' ', end='\n')
                continue
            
            GroundTruth = np.ones(np.shape(Predictions)[0])*clNum
            PtdLabels = np.argmax(Predictions, axis=1)
            # PtdLabels, label_counts = scipy.stats.mode(PtdLabels_temp, axis=1)
            # PtdLabels = np.array(PtdLabels.flatten())
            # print('PtdLabels: ', np.shape(PtdLabels), ' GroundTruth: ', np.shape(GroundTruth))
            
            print(np.shape(Predictions), ' acc=', np.round(np.sum(PtdLabels==GroundTruth)*100/np.size(GroundTruth), 2), end='\n')
            if np.size(PtdLabels_Ensemble)<=1:
                PtdLabels_Ensemble = PtdLabels
                GroundTruth_Ensemble = GroundTruth
                Predictions_Ensemble = Predictions
            else:
                PtdLabels_Ensemble = np.append(PtdLabels_Ensemble, PtdLabels)
                GroundTruth_Ensemble = np.append(GroundTruth_Ensemble, GroundTruth)
                Predictions_Ensemble = np.append(Predictions_Ensemble, Predictions, axis=0)

    testingTimeTaken = time.clock() - start
    print('\t\t\t', PARAMS['fold'], ' Time taken for model testing: ',testingTimeTaken)
    ConfMat_Ensemble, fscore_Ensemble = misc.getPerformance(PtdLabels_Ensemble, GroundTruth_Ensemble)
    accuracy_Ensemble = np.round(np.sum(PtdLabels_Ensemble==GroundTruth_Ensemble)*100/len(GroundTruth_Ensemble), 4)
    
    for featName in Ensemble_Train_Params.keys():
        # print(featName, 'individual_performances: ', np.shape(individual_performances[featName]['PtdLabels']), np.shape(GroundTruth_Ensemble))
        indv_PtdLabels = np.argmax(individual_performances[featName]['Predictions'], axis=1)
        ConfMat_indv, fscore_indv = misc.getPerformance(indv_PtdLabels, individual_performances[featName]['GroundTruth'])
        individual_performances[featName]['fscore'] = fscore_indv

    Ensemble_Test_Params = {
        'loss': -1,
        'accuracy_Ensemble': accuracy_Ensemble,
        'testingTimeTaken': testingTimeTaken,
        'ConfMat_Ensemble': ConfMat_Ensemble,
        'fscore_Ensemble': fscore_Ensemble,
        'PtdLabels_Ensemble': PtdLabels_Ensemble,
        'Predictions_Ensemble': Predictions_Ensemble,
        'GroundTruth_Ensemble': GroundTruth_Ensemble,
        'individual_performances': individual_performances,
        }

    return Ensemble_Test_Params




def generator_test_noise(PARAMS, featName, file_name, clNum, targetdB):
    fName = PARAMS['test_folder'] + '/' + featName + '/'+ PARAMS['classes'][clNum] + '/' + file_name
    batchData = np.load(fName, allow_pickle=True).item()[targetdB]

    # print('generator_test_noise: ', featName, PARAMS['input_shape'][featName])
    if 'Ensemble' in PARAMS['clFunc']:
        batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'][featName])
    else:
        batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])

    batchData = np.expand_dims(batchData, axis=3)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel





def test_cnn_noise(PARAMS, Train_Params):
    start = time.clock()
    GroundTruth = []
    Predictions = {}
    PtdLabels = {}
    for dB in PARAMS['noise_dB_range']:
        Predictions[dB] = np.empty([])
        PtdLabels[dB] = []
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    
    for classname in PARAMS['test_files'].keys():
        clNum = class_labels[classname]
        files = PARAMS['test_files'][classname]
        # print('test_files: ', files)
        fl_count = 0
        for fl in files:
            fl_count += 1
            print('\t\t\t',PARAMS['fold'], PARAMS['classes'][clNum], fl, fl_count, '/', len(files))
            count += 1
            numFV = 0
            if not os.path.exists(PARAMS['test_folder'] + '/' + PARAMS['featName'] + '/'+ classname + '/' + fl):
                continue
            for targetdB in PARAMS['noise_dB_range']:
                batchData, batchLabel = generator_test_noise(PARAMS, PARAMS['featName'], fl, clNum, targetdB)
                pred = Train_Params['model'].predict(x=batchData)
                numFV = np.shape(pred)[0]
                if np.size(Predictions[targetdB])<=1:
                    Predictions[targetdB] = np.array(pred, ndmin=2)
                    PtdLabels[targetdB].extend(np.argmax(pred, axis=1).tolist())
                else:
                    Predictions[targetdB] = np.append(Predictions[targetdB], np.array(pred, ndmin=2), axis=0)
                    PtdLabels[targetdB].extend(np.argmax(pred, axis=1).tolist())
                print('\t\t\t\t dB=',targetdB,' batchData: ', np.shape(batchData), np.shape(Predictions[targetdB]), ' acc=', np.round(np.sum(np.argmax(pred, axis=1)==np.array([clNum]*numFV))*100/numFV, 2))
            GroundTruth.extend([clNum]*numFV)
    
    testingTimeTaken = time.clock() - start
    print('\t\t\t', PARAMS['fold'], ' Time taken for model testing: ', testingTimeTaken)
    
    ConfMat = {}
    fscore = {}
    accuracy = {}
    for dB in PARAMS['noise_dB_range']:
        ConfMat_dB, fscore_dB = misc.getPerformance(PtdLabels[dB], GroundTruth)
        ConfMat_dB = np.reshape(ConfMat_dB, (len(PARAMS['classes']),len(PARAMS['classes'])))
        accuracy_dB = np.round(np.sum(np.diag(ConfMat_dB))/np.sum(ConfMat_dB), 4)
        ConfMat[dB] = ConfMat_dB
        fscore[dB] = fscore_dB
        accuracy[dB] = accuracy_dB
    

    Test_Params_Noise = {
        'loss': -1,
        'accuracy': accuracy,
        'testingTimeTaken': testingTimeTaken,
        'ConfMat': ConfMat,
        'fscore': fscore,
        'PtdLabels': PtdLabels,
        'Predictions': Predictions,
        'GroundTruth': GroundTruth,
        }

    return Test_Params_Noise





def test_cnn_ensemble_noise(PARAMS, Ensemble_Train_Params):
    start = time.clock()
    GroundTruth_Ensemble = {dB:[] for dB in PARAMS['noise_dB_range']}
    Predictions_Ensemble = {dB:np.empty([]) for dB in PARAMS['noise_dB_range']}
    PtdLabels_Ensemble = {dB:[] for dB in PARAMS['noise_dB_range']}
    count = -1
    class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
    basic_storage_cell = {'Predictions':np.empty([]), 'GroundTruths':np.empty([]), 'fscore': [0, 0, 0]}
    individual_performances = {
        'Khonglah_et_al': {db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        'Sell_et_al':{db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        'MFCC-39':{db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        'Melspectrogram':{db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        'HNGDMFCC':{db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        'MGDCC':{db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        'IFCC':{db:basic_storage_cell for db in PARAMS['noise_dB_range']}, 
        }
    temp_folder = PARAMS['opDir'] + '/__temp/'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    for classname in PARAMS['test_files'].keys():
        clNum = class_labels[classname]
        files = PARAMS['test_files'][classname]
        # print('test_files: ', files)
        fl_count = 0
        for fl in files:
            fl_count += 1
            print('\t\t\t',PARAMS['fold'], PARAMS['classes'][clNum], fl, fl_count, '/', len(files), end='\n')
            count += 1
            PtdLabels = None
            GroundTruth = np.empty([])
            Predictions = {dB:np.empty([]) for dB in PARAMS['noise_dB_range']}
            PtdLabels = {dB:[] for dB in PARAMS['noise_dB_range']}
            empty_predictions = False
            for featName in Ensemble_Train_Params.keys():
                empty_predictions = False
                curr_fName = PARAMS['test_folder'] + '/' + featName + '/'+ classname + '/' + fl
                if not os.path.exists(curr_fName):
                    empty_predictions = True
                    break

                Train_Params = Ensemble_Train_Params[featName]
                batchData = None
                batchLabel = None
                if not os.path.exists(PARAMS['test_folder'] + '/' + featName + '/'+ classname + '/' + fl):
                    continue
                for targetdB in PARAMS['noise_dB_range']:
                    temp_file = temp_folder + '/pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + str(targetdB) + 'dB_' + fl.split('.')[0] + '.pkl'
                    if not os.path.exists(temp_file):
                        batchData, batchLabel = generator_test_noise(PARAMS, featName, fl, clNum, targetdB)
                        pred = Train_Params['model'].predict(x=batchData)
                        misc.save_obj(pred, temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + str(targetdB) + 'dB_' + fl.split('.')[0])
                    else:
                        pred = misc.load_obj(temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + str(targetdB) + 'dB_' + fl.split('.')[0])
                    
                    if np.size(individual_performances[featName][targetdB]['Predictions'])<=1:
                        individual_performances[featName][targetdB]['Predictions'] = np.array(pred, ndmin=2)
                        individual_performances[featName][targetdB]['GroundTruth'] = np.ones(np.shape(pred)[0])*clNum
                    else:
                        individual_performances[featName][targetdB]['Predictions'] = np.append(individual_performances[featName][targetdB]['Predictions'], np.array(pred, ndmin=2), axis=0)
                        individual_performances[featName][targetdB]['GroundTruth'] = np.append(individual_performances[featName][targetdB]['GroundTruth'], np.ones(np.shape(pred)[0])*clNum)
    
                    if np.size(Predictions[targetdB])<=1:
                        Predictions[targetdB] = np.array(pred, ndmin=2)
                    else:
                        empty_predictions = False
                        if np.shape(pred)[0]!=np.shape(Predictions[targetdB])[0]:
                            if np.shape(pred)[0]>np.shape(Predictions[targetdB])[0]:
                                pred = pred[np.shape(Predictions[targetdB])[0], :]
                            else:
                                empty_predictions = True
                                break
                        Predictions[targetdB] = np.add(Predictions[targetdB], np.array(pred, ndmin=2))
            
            if empty_predictions:
                print(' ', end='\n')
                continue
            
            for dB in PARAMS['noise_dB_range']:
                GroundTruth = np.array(np.ones(np.shape(Predictions[dB])[0])*clNum, ndmin=2).T
                PtdLabels[dB] = np.array(np.argmax(Predictions[dB], axis=1), ndmin=2).T
                # print('PtdLabels[dB]: ', np.shape(PtdLabels[dB]), np.shape(GroundTruth), np.sum(PtdLabels[dB]==GroundTruth), np.shape(GroundTruth)[0])
                print('\t\t\t\t', dB, 'dB\t', np.shape(Predictions[dB]), ' acc=', np.round(np.sum(PtdLabels[dB]==GroundTruth)*100/np.shape(GroundTruth)[0], 2), end='\n')
                if np.size(PtdLabels_Ensemble[dB])<=1:
                    PtdLabels_Ensemble[dB] = PtdLabels[dB]
                    Predictions_Ensemble[dB] = Predictions[dB]
                    GroundTruth_Ensemble[dB] = GroundTruth
                else:
                    PtdLabels_Ensemble[dB] = np.append(PtdLabels_Ensemble[dB], PtdLabels[dB])
                    Predictions_Ensemble[dB] = np.append(Predictions_Ensemble[dB], Predictions[dB], axis=0)
                    GroundTruth_Ensemble[dB] = np.append(GroundTruth_Ensemble[dB], GroundTruth)

    testingTimeTaken = time.clock() - start
    print('\t\t\t', PARAMS['fold'], ' Time taken for model testing: ',testingTimeTaken)
    ConfMat_Ensemble = {}
    fscore_Ensemble = {}
    accuracy_Ensemble = {}
    for dB in PARAMS['noise_dB_range']:
        # print(dB, np.shape(PtdLabels_Ensemble[dB]), np.shape(GroundTruth_Ensemble[dB]))
        ConfMat_dB, fscore_dB = misc.getPerformance(PtdLabels_Ensemble[dB], GroundTruth_Ensemble[dB])
        ConfMat_dB = np.reshape(ConfMat_dB, (len(PARAMS['classes']),len(PARAMS['classes'])))
        accuracy_dB = np.round(np.sum(np.diag(ConfMat_dB))/np.sum(ConfMat_dB), 4)
        ConfMat_Ensemble[dB] = ConfMat_dB
        fscore_Ensemble[dB] = fscore_dB
        accuracy_Ensemble[dB] = accuracy_dB
    
    for featName in Ensemble_Train_Params.keys():
        for dB in PARAMS['noise_dB_range']:
            indv_PtdLabels_dB = np.argmax(individual_performances[featName][dB]['Predictions'], axis=1)
            ConfMat_indv_dB, fscore_indv_dB = misc.getPerformance(indv_PtdLabels_dB, individual_performances[featName][dB]['GroundTruth'])
            individual_performances[featName][dB]['fscore'] = fscore_indv_dB

    Ensemble_Test_Params = {
        'loss': -1,
        'accuracy_Ensemble': accuracy_Ensemble,
        'testingTimeTaken': testingTimeTaken,
        'ConfMat_Ensemble': ConfMat_Ensemble,
        'fscore_Ensemble': fscore_Ensemble,
        'PtdLabels_Ensemble': PtdLabels_Ensemble,
        'Predictions_Ensemble': Predictions_Ensemble,
        'GroundTruth_Ensemble': GroundTruth_Ensemble,
        'individual_performances': individual_performances,
        }

    return Ensemble_Test_Params




def train_cnn_transfer_learn(PARAMS, Frozen_Model_Params):
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
        new_model = Frozen_Model_Params['new_model']
        print_model_summary(arch_file, new_model)

        model, trainingTimeTaken, History = train_model(PARAMS, new_model, weightFile)
        
        if PARAMS['save_flag']:
            model.save_weights(weightFile) # Save the weights
            with open(architechtureFile, 'w') as f: # Save the model architecture
                f.write(model.to_json())
            np.savez(paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], input_shape=PARAMS['input_shape'], lr=Frozen_Model_Params['learning_rate'], trainingTimeTaken=trainingTimeTaken)
        print('CNN model trained.')
    else:
        PARAMS['epochs'] = np.load(paramFile)['epochs']
        PARAMS['batch_size'] = np.load(paramFile)['batch_size']
        PARAMS['input_shape'] = np.load(paramFile)['input_shape']
        Frozen_Model_Params['learning_rate'] = np.load(paramFile)['lr']
        trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
        optimizer = optimizers.Adam(lr=Frozen_Model_Params['learning_rate'])
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
            'learning_rate': Frozen_Model_Params['learning_rate'],
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params
