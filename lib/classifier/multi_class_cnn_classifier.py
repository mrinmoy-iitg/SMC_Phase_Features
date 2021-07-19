#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:02:10 2020

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
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import PatchExtractor




def get_feature_patches(FV, patch_size, patch_shift, input_shape):
    FV = StandardScaler(copy=False).fit_transform(FV)
    # FV should be of the shape (nFeatures, nFrames)
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV.T
    patches = np.empty([])

    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)

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
        first_7_cep_dim = np.array(list(range(0,7))+list(range(13,20))+list(range(26,33)))
        patches = patches[:, first_7_cep_dim, :]
    # print('Patches: ', np.shape(patches))
    
    return patches



def generator(PARAMS, folder, file_list, batchSize):
    batch_count = 0
    batchData_classwise = {}
    balance = {}
    file_list_temp = {}
    for clNum in PARAMS['classes'].keys():
        np.random.shuffle(file_list[PARAMS['classes'][clNum]])
        file_list_temp[clNum] = file_list[PARAMS['classes'][clNum]].copy()
        batchData_classwise[clNum] = np.empty([], dtype=float)
        balance[clNum] = 0
    while 1:
        cl_count = 0
        batchData = np.empty([], dtype=float)
        batchLabel = np.empty([], dtype=float)
        for clNum in PARAMS['classes'].keys():
            while balance[clNum]<batchSize:
                if not file_list_temp[clNum]:
                    file_list_temp[clNum] = file_list[PARAMS['classes'][clNum]].copy()
                fName = file_list_temp[clNum].pop()
                fName_path = folder + '/' + PARAMS['featName'] + '/' + PARAMS['classes'][clNum] + '/' + fName
                if not os.path.exists(fName_path):
                    continue
                # print('fName_path: ', fName_path)
                fv = np.load(fName_path, allow_pickle=True)
                fv = get_feature_patches(fv, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
                fv = np.expand_dims(fv, axis=3)
                if balance[clNum]==0:
                    batchData_classwise[clNum] = fv
                else:
                    batchData_classwise[clNum] = np.append(batchData_classwise[clNum], fv, axis=0)
                balance[clNum] += np.shape(fv)[0]
                # print(PARAMS['classes'][clNum], ': ', batchSize, balance[clNum], np.shape(batchData_classwise[clNum]))
            
            if cl_count==0:
                batchData = batchData_classwise[clNum][:batchSize, :]
                batchLabel = np.ones(batchSize)*clNum
            else:
                batchData = np.append(batchData, batchData_classwise[clNum][:batchSize, :], axis=0)
                batchLabel = np.append(batchLabel, np.ones(batchSize)*clNum)
            
            cl_count += 1
            balance[clNum] -= batchSize
            batchData_classwise[clNum] = batchData_classwise[clNum][batchSize:, :]

        OHE_batchLabel = to_categorical(batchLabel, num_classes=len(PARAMS['classes']))                
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
    
    input_img = Input(input_shape)
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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])    

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
        



def generator_test(PARAMS, featName, file_name, clNum):
    fName = PARAMS['test_folder'] + '/' + featName + '/'+ PARAMS['classes'][clNum] + '/' + file_name
    # startTime = time.clock()
    batchData = np.load(fName, allow_pickle=True)
    # print('Loading: ', time.clock()-startTime)

    # startTime = time.clock()
    batchData = get_feature_patches(batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])
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
    loss, performance = Train_Params['model'].evaluate(
            generator(PARAMS, PARAMS['test_folder'], PARAMS['test_files'], PARAMS['batch_size']),
            steps=PARAMS['test_steps'], 
            verbose=1
            )
    print('loss: ', loss, ' performance: ', performance)
    testingTimeTaken = time.clock() - start
    print('Time taken for model testing: ',testingTimeTaken)
    
    return loss, performance, testingTimeTaken


    
def test_model(PARAMS, Train_Params):
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
            pred = Train_Params['model'].predict(x=batchData)
            pred_lab = np.argmax(pred, axis=1)
            PtdLabels.extend(pred_lab)
            GroundTruth.extend(batchLabel.tolist())
            if np.size(Predictions)<=1:
                Predictions = pred
            else:
                Predictions = np.append(Predictions, pred, 0)
            print(PARAMS['classes'][clNum], fl, np.shape(batchData), ' acc=', np.round(np.sum(pred_lab==batchLabel)*100/len(batchLabel), 2))

    testingTimeTaken = time.clock() - startTime
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
                temp_file = temp_folder + '/pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + fl.split('.')[0] + '.pkl'
                if not os.path.exists(temp_file):
                    batchData, batchLabel = generator_test(PARAMS, featName, fl, clNum)
                    pred = Train_Params['model'].predict(x=batchData)
                    misc.save_obj(pred, temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + fl.split('.')[0])
                else:
                    pred = misc.load_obj(temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + fl.split('.')[0])
                # print('indv_labels: ', np.shape(indv_labels), np.shape(individual_performances[featName]['PtdLabels']))
                if np.size(individual_performances[featName]['Predictions'])<=1:
                    individual_performances[featName]['Predictions'] = np.array(pred, ndmin=2)
                    individual_performances[featName]['GroundTruth'] = np.ones(np.shape(pred)[0])*clNum
                else:
                    individual_performances[featName]['Predictions'] = np.append(individual_performances[featName]['Predictions'], np.array(pred, ndmin=2), axis=0)
                    individual_performances[featName]['GroundTruth'] = np.append(individual_performances[featName]['GroundTruth'], np.ones(np.shape(pred)[0])*clNum)

                if np.size(Predictions)<=1:
                    Predictions = np.array(pred, ndmin=2)
                else:
                    empty_predictions = False
                    if np.shape(pred)[0]!=np.shape(Predictions)[0]:
                        if np.shape(pred)[0]>np.shape(Predictions)[0]:
                            pred = pred[np.shape(Predictions)[0], :]
                        else:
                            empty_predictions = True
                            break
                    Predictions = np.add(Predictions, np.array(pred, ndmin=2))
            
            if empty_predictions:
                print(' ', end='\n')
                continue
            
            GroundTruth = np.ones(np.shape(Predictions)[0])*clNum
            PtdLabels = np.argmax(Predictions, axis=1)
            print(np.shape(Predictions), ' acc=', np.round(np.sum(PtdLabels==GroundTruth)*100/len(GroundTruth), 2), end='\n')
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
