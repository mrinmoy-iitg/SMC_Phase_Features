#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:43:57 2018

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import lib.misc as misc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import PatchExtractor
from tensorflow.keras.utils import to_categorical





def get_feature_patches(PARAMS, FV, patch_size, patch_shift, input_shape):
    FV = StandardScaler(copy=False).fit_transform(FV)
    # FV should be of the shape (nFeatures, nFrames)
    if any(np.array([9,10,21,22,39])==np.shape(FV)[1]): 
        FV = FV.T
    patches = np.empty([])

    if np.shape(FV)[1]<patch_size:
        # print('Size append: ', np.shape(FV), patch_size)
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)

    numPatches = int(np.ceil(np.shape(FV)[1]/patch_shift))
    patches = PatchExtractor(patch_size=(np.shape(FV)[0], patch_size), max_patches=numPatches).transform(np.expand_dims(FV, axis=0))


    patches_mean = np.mean(patches, axis=2)
    patches_var = np.var(patches, axis=2)
    patches_mean_var = np.append(patches_mean, patches_var, axis=1)
    # print('sklearn splitting: ', time.clock()-startTime, np.shape(patches))


    # print('Patches: ', np.shape(patches))
    if np.shape(patches_mean_var)[1]==44:
        patches_mean_var = patches_mean_var[:,list(range(0,21))+list(range(22,43))]
    elif np.shape(patches_mean_var)[1]==78:
        if not PARAMS['39_dim_CC_feat']:
            first_7_cep_dim = np.array(list(range(0,7))+list(range(13,20))+list(range(26,33))+list(range(39,46))+list(range(52,59))+list(range(65,72)))
            patches_mean_var = patches_mean_var[:, first_7_cep_dim]
    # print('patches_mean_var: ', np.shape(patches_mean_var))
    
    return patches_mean_var



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
            # print('sp_fName_path: ', sp_fName_path)
            if not os.path.exists(sp_fName_path):
                continue
            # print('sp_fName_path: ', sp_fName_path)
            fv_sp = np.load(sp_fName_path, allow_pickle=True)
            fv_sp = get_feature_patches(PARAMS, fv_sp, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
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
            if balance_mu==0:
                batchData_mu = fv_mu
            else:
                batchData_mu = np.append(batchData_mu, fv_mu, axis=0)
            balance_mu += np.shape(fv_mu)[0]
            # print('Music: ', batchSize, balance_mu, np.shape(batchData_mu))
                
        batchData = batchData_sp[:batchSize, :]
        batchData = np.append(batchData, batchData_mu[:batchSize, :], axis=0)
        # batchData = StandardScaler(copy=False).fit_transform(batchData)
        
        balance_sp -= batchSize
        balance_mu -= batchSize
        batchData_sp = batchData_sp[batchSize:, :]
        batchData_mu = batchData_mu[batchSize:, :]            

        class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        batchLabel = np.ones(2*batchSize)
        batchLabel[:batchSize] = class_labels['speech']
        batchLabel[batchSize:] = class_labels['music']
        if len(PARAMS['classes'])>2:
            batchLabel = to_categorical(batchLabel, num_classes=len(PARAMS['classes']))
                
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData))
        yield batchData, batchLabel






def train_model(PARAMS, train_data, train_label, model, epochs, batch_size, weightFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    logFile = '/'.join(weightFile.split('/')[:-2]) + '/log_fold' + str(PARAMS['fold']) + '.csv'
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.clock()

    if not PARAMS['data_generator']    :
        # Train the model
        History = model.fit(
                x=train_data,
                y=train_label, 
                epochs=epochs,
                batch_size=batch_size, 
                verbose=1,
                validation_split=0.3,
                callbacks=[csv_logger, es, mcp],
                shuffle=True,
                )
    else:
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



# define base model
def dnn_model(input_dim, output_dim, num_dnn_lyr, num_dnn_nodes):
    # create model
    
    input_layer = Input(input_dim)
    
    x = Dense(num_dnn_nodes, kernel_regularizer=l2())(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    for i in range(1, num_dnn_lyr):
        x = Dense(num_dnn_nodes, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)

    if output_dim>2:
        output_layer = Dense(output_dim, kernel_regularizer=l2())(x)
        output_layer = Activation('softmax')(output_layer)
        loss_type = 'categorical_crossentropy'
    else:
        output_layer = Dense(1, kernel_regularizer=l2())(x)
        output_layer = Activation('sigmoid')(output_layer)
        loss_type = 'binary_crossentropy'

    model = Model(input_layer, output_layer)
    
    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)

    optimizerName = 'Adam'
    # Compile model
    model.compile(loss=loss_type, optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    return model, optimizerName, learning_rate




def print_model_summary(arch_file, model):
    stringlist = ['Architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n']
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    with open(arch_file, 'w+', encoding='utf8') as f:
        f.write(short_model_summary)




'''
This function is the driver function for learning and evaluating a DNN model. 
'''
def train_dnn(PARAMS, train_data, train_label, num_dnn_lyr, num_dnn_nodes):
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
    
    output_dim = len(PARAMS['classes'])
    print(output_dim)

    print('Weight file: ', weightFile, PARAMS['input_dim'], output_dim)

    if not os.path.exists(paramFile):
        model, optimizerName, learning_rate = dnn_model(PARAMS['input_dim'], output_dim, num_dnn_lyr, num_dnn_nodes)
        print_model_summary(arch_file, model)
            
        model, trainingTimeTaken, History = train_model(PARAMS, train_data, train_label, model, PARAMS['epochs'], PARAMS['batch_size'], weightFile)
        if PARAMS['save_flag']:
            model.save_weights(weightFile)
            with open(architechtureFile, 'w') as f:
                f.write(model.to_json())
            np.savez(paramFile, ep=str(PARAMS['epochs']), bs=str(PARAMS['batch_size']), lr=str(learning_rate), TTT=str(trainingTimeTaken))
    else:
        PARAMS['epochs'] = int(np.load(paramFile)['ep'])
        PARAMS['batch_size'] = int(np.load(paramFile)['bs'])
        learning_rate = float(np.load(paramFile)['lr'])
        trainingTimeTaken = float(np.load(paramFile)['TTT'])
        optimizerName = 'Adam'

        # with open(architechtureFile, 'r') as f:
        #     model = model_from_json(f.read())
        try:
            with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
                model = model_from_json(f.read())
        except:
            num_dnn_lyr = PARAMS['DNN_optimal_params'][PARAMS['dataset_name']][PARAMS['featName']]['hidden_lyrs']
            num_dnn_nodes = PARAMS['DNN_optimal_params'][PARAMS['dataset_name']][PARAMS['featName']]['hidden_nodes']
            model, optimizerName, learning_rate_temp = dnn_model(PARAMS['input_dim'], 1, num_dnn_lyr, num_dnn_nodes)

        model.load_weights(weightFile)
        opt = optimizers.Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        print('DNN model exists! Loaded. Training time required=',trainingTimeTaken)
        print(model.summary())
    
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': PARAMS['epochs'],
            'batch_size': PARAMS['batch_size'],
            'learning_rate': learning_rate,
            'optimizerName': optimizerName,
            'paramFile': paramFile,
            'architechtureFile': architechtureFile,
            'weightFile': weightFile,
            }
    
    return Train_Params




def generator_test(PARAMS, featName, file_name, clNum):
    fName = PARAMS['test_folder'] + '/' + featName + '/'+ PARAMS['classes'][clNum] + '/' + file_name
    if not os.path.exists(fName):
        return [], []
    batchData = np.load(fName, allow_pickle=True)
    batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])
    # batchData = StandardScaler(copy=False).fit_transform(batchData)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel




def test_model(PARAMS, test_data, test_label, Train_Params):
    loss = 0
    performance = 0
    testingTimeTaken = 0
    PtdLabels = []
    
    start = time.clock()
    if not PARAMS['data_generator']:
        loss, performance = Train_Params['model'].evaluate(x=test_data, y=test_label)
        Predictions = Train_Params['model'].predict(test_data)
        PtdLabels = np.array(Predictions>0.5).astype(int)
        GroundTruth = test_label
        testingTimeTaken = time.clock() - start
        print('Time taken for model testing: ',testingTimeTaken)
        ConfMat, fscore = misc.getPerformance(PtdLabels, GroundTruth)
    else:
        loss, performance = Train_Params['model'].evaluate_generator(
            generator(PARAMS, PARAMS['test_folder'], PARAMS['test_files'], PARAMS['batch_size']),
            steps=PARAMS['test_steps'],
            verbose=1,
            )
        print('loss: ', loss, ' performance: ', performance)

        PtdLabels = []
        GroundTruth = []
        Predictions = np.empty([])
        count = -1
        class_labels = {PARAMS['classes'][key]:key for key in PARAMS['classes'].keys()}
        # startTime = time.clock()
        for classname in PARAMS['test_files'].keys():
            clNum = class_labels[classname]
            files = PARAMS['test_files'][classname]
            # print('test_files: ', files)
            for fl in files:
                count += 1
                batchData, batchLabel = generator_test(PARAMS, PARAMS['featName'], fl, clNum)
                if batchData==[]:
                    continue
                # endTime = time.clock()
                # print('Data loading time: ', endTime-startTime)
                
                # startTime = time.clock()
                pred = Train_Params['model'].predict(x=batchData)
                # print('Prediction time: ', time.clock()-startTime, np.shape(pred))
    
                if len(PARAMS['classes'])>2:
                    pred_lab = np.argmax(pred, axis=1)
                else:
                    pred_lab = np.squeeze(np.array(np.array(pred)>0.5).astype(int))
                # print(clNum, ' batchLabel: ', batchLabel)
                # print(clNum, ' pred_lab: ', pred_lab)
                PtdLabels.extend(pred_lab)
                GroundTruth.extend(batchLabel.tolist())
                # print('pred_lab: ', np.sum(pred_lab==0), np.sum(pred_lab==1))
                # print('ground_truth: ', np.sum(batchLabel==0), np.sum(batchLabel==1))
                if np.size(Predictions)<=1:
                    Predictions = pred
                else:
                    Predictions = np.append(Predictions, pred, 0)
                print(PARAMS['classes'][clNum], fl, np.shape(batchData), ' acc=', np.round(np.sum(pred_lab==batchLabel)*100/len(batchLabel), 2))
    
        testingTimeTaken = time.clock() - start
        print('Time taken for model testing: ',testingTimeTaken)
        ConfMat, fscore = misc.getPerformance(PtdLabels, GroundTruth)
    
    return loss, performance, testingTimeTaken, ConfMat, fscore, PtdLabels, Predictions, GroundTruth




def test_dnn(PARAMS, test_data, test_label, Train_Params):
    loss, performance, testingTimeTaken, ConfMat, fscore, PtdLabels, Predictions, GroundTruth = test_model(PARAMS, test_data, test_label, Train_Params)

    loss, performance, testingTimeTaken, ConfMat, fscore, PtdLabels, Predictions, GroundTruth = test_model(PARAMS, test_data, test_label, Train_Params)
    
    print('loss: ', loss)
    print('performancermance: ', performance)

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
    batchData = np.load(fName, allow_pickle=True)
    batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'][featName])
    # batchData = StandardScaler(copy=False).fit_transform(batchData)
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel




def test_dnn_ensemble(PARAMS, Ensemble_Train_Params):
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
                    batchData, batchLabel = generator_test_ensemble(PARAMS, featName, fl, clNum)
                    pred = Train_Params['model'].predict(x=batchData)
                    misc.save_obj(pred, temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + fl.split('.')[0])
                else:
                    pred = misc.load_obj(temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + fl.split('.')[0])
                # print('indv_labels: ', np.shape(indv_labels), np.shape(individual_performances[featName]['PtdLabels']))
                if np.size(individual_performances[featName]['Predictions'])<=1:
                    individual_performances[featName]['Predictions'] = np.array(pred)
                    individual_performances[featName]['GroundTruth'] = np.ones(np.shape(pred)[0])*clNum
                else:
                    individual_performances[featName]['Predictions'] = np.append(individual_performances[featName]['Predictions'], np.array(pred), axis=0)
                    individual_performances[featName]['GroundTruth'] = np.append(individual_performances[featName]['GroundTruth'], np.ones(np.shape(pred)[0])*clNum)

                if np.size(Predictions)<=1:
                    Predictions = np.array(pred)
                else:
                    # print('Predictions: ', np.shape(Predictions), np.shape(pred), np.shape(Predictions)[0], np.shape(pred)[0])
                    empty_predictions = False
                    if np.shape(pred)[0]!=np.shape(Predictions)[0]:
                        # print('Predictions size mismatch ', featName, np.shape(Predictions), np.shape(pred))
                        if np.shape(pred)[0]>np.shape(Predictions)[0]:
                            pred = pred[:np.shape(Predictions)[0], :]
                        else:
                            empty_predictions = True
                            break
                            
                    Predictions = np.add(Predictions, np.array(pred))

            # print('empty_predictions: ', empty_predictions)
            if empty_predictions:
                print(' ', end='\n')
                continue
            
            Predictions /= len(Ensemble_Train_Params)
            GroundTruth = np.ones(np.shape(Predictions)[0])*clNum
            PtdLabels = np.array(Predictions>0.5).flatten().astype(int)
            # print('PtdLabels: ', np.shape(PtdLabels), PtdLabels)
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




def generator_test_noise(PARAMS, featName, file_name, clNum, targetdB):
    fName = PARAMS['test_folder'] + '/' + featName + '/'+ PARAMS['classes'][clNum] + '/' + file_name
    batchData = np.load(fName, allow_pickle=True).item()[targetdB]
    
    if 'Ensemble' in PARAMS['clFunc']:
        batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'][featName])
    else:
        batchData = get_feature_patches(PARAMS, batchData, PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift_test'], PARAMS['input_shape'])        
    numLab = np.shape(batchData)[0]
    batchLabel = np.array([clNum]*numLab)

    return batchData, batchLabel





def test_dnn_noise(PARAMS, Train_Params):
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
                numFV = len(pred)
                if np.size(Predictions[targetdB])<=1:
                    Predictions[targetdB] = np.array(pred)
                    PtdLabels[targetdB].extend(np.array(pred>0.5).astype(int).tolist())
                else:
                    Predictions[targetdB] = np.append(Predictions[targetdB], np.array(pred), axis=0)
                    PtdLabels[targetdB].extend(np.array(pred>0.5).astype(int).tolist())
                
                print('\t\t\t\t dB=',targetdB,' batchData: ', np.shape(batchData), np.shape(Predictions[targetdB]), ' acc=', np.round(np.sum(np.squeeze(np.array(pred>0.5).astype(int)) == np.array([clNum]*numFV))*100/numFV, 2))
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



def test_dnn_ensemble_noise(PARAMS, Ensemble_Train_Params):
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
                        # print('batchData: ', batchData)
                        misc.save_obj(pred, temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + str(targetdB) + 'dB_' + fl.split('.')[0])
                    else:
                        pred = misc.load_obj(temp_folder, 'pred_' + classname + '_fold' + str(PARAMS['fold']) + '_' + featName + '_' + str(targetdB) + 'dB_' + fl.split('.')[0])

                    if np.size(individual_performances[featName][targetdB]['Predictions'])<=1:
                        individual_performances[featName][targetdB]['Predictions'] = np.array(pred)
                        individual_performances[featName][targetdB]['GroundTruth'] = np.ones(np.shape(pred)[0])*clNum
                    else:
                        individual_performances[featName][targetdB]['Predictions'] = np.append(individual_performances[featName][targetdB]['Predictions'], np.array(pred), axis=0)
                        individual_performances[featName][targetdB]['GroundTruth'] = np.append(individual_performances[featName][targetdB]['GroundTruth'], np.ones(np.shape(pred)[0])*clNum)
    
                    if np.size(Predictions[targetdB])<=1:
                        Predictions[targetdB] = np.array(pred)
                    else:
                        empty_predictions = False
                        if np.shape(pred)[0]!=np.shape(Predictions[targetdB])[0]:
                            if np.shape(pred)[0]>np.shape(Predictions[targetdB])[0]:
                                pred = pred[np.shape(Predictions[targetdB])[0], :]
                            else:
                                empty_predictions = True
                                break
                        Predictions[targetdB] = np.add(Predictions[targetdB], np.array(pred))
            
            if empty_predictions:
                print(' ', end='\n')
                continue
            
            # print('GroundTruth: ', np.shape(GroundTruth))
            for dB in PARAMS['noise_dB_range']:
                GroundTruth = np.array(np.ones(np.shape(Predictions[dB])[0])*clNum, ndmin=2).T
                Predictions[dB] /= len(Ensemble_Train_Params)
                PtdLabels[dB] = np.array(Predictions[dB]>0.5).astype(int)
                # print('PtdLabels[dB]: ', np.shape(PtdLabels[dB]), np.shape(GroundTruth), np.sum(PtdLabels[dB]==GroundTruth))
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
