#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:55:09 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""


import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import PatchExtractor
import time
import textgrids



def save_obj(obj, folder, name):
    with open(folder+'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    with open(folder+'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



def pca_dim_reduction(train_data, test_data, n_components=0.95):
    print('Before PCA: ', np.shape(train_data), np.shape(test_data))
    dim_reduction = PCA(n_components, whiten=False)
    dim_reduction.fit(train_data)
    train_data = dim_reduction.transform(train_data)
    test_data = dim_reduction.transform(test_data)
    print('After PCA: ', np.shape(train_data), np.shape(test_data))
    return train_data, test_data
    



def scale_data(train_data, test_data):
    print('Scale data func: train_data=', np.shape(train_data))
    std_scale = StandardScaler().fit(train_data)
    train_data_scaled = std_scale.transform(train_data)
    test_data_scaled = std_scale.transform(test_data)

    return train_data_scaled, test_data_scaled



def preprocess_data(PARAMS, train_data, train_label, test_data):
    if PARAMS['data_balancing']:
        from imblearn.combine import SMOTEENN
        print('Unbalanced data: ', np.shape(train_data))
        # Over and under sampling
        smote_enn = SMOTEENN(sampling_strategy=1.0)
        train_data, train_label = smote_enn.fit_resample(train_data, train_label)
        print('Balanced data: ', np.shape(train_data))
    
    if PARAMS['PCA_flag']:
        train_data, test_data = pca_dim_reduction(train_data, test_data)

    if PARAMS['scale_data']:
        train_data, test_data = scale_data(train_data, test_data)
    
    return train_data, train_label, test_data
    


def create_CV_folds(folder, featName, classes, cv=3):
    if not os.path.exists(folder+'/cv_file_list.pkl'):
        cv_file_list = {}
        for clNum in classes.keys():
            path = folder + '/' + featName + '/' + classes[clNum] + '/'
            files = np.array(os.listdir(path))
            np.random.shuffle(files)
            files_per_fold = int(np.ceil(len(files)/cv))
            cv_file_list[classes[clNum]] = {}
            fl_count = 0
            for cv_num in range(cv):
                cv_file_list[classes[clNum]]['fold'+str(cv_num)] = files[fl_count:np.min([fl_count+files_per_fold, len(files)])]
                fl_count += files_per_fold
            
        save_obj(cv_file_list, folder, 'cv_file_list')
        print('CV folds created')
    else:
        cv_file_list = load_obj(folder, 'cv_file_list')
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

    
    
    
def load_data_from_files(classes, folder, featName, files, patch_size, patch_shift, input_shape):
    label = []
    data = np.empty([])
    for clNum in classes.keys():
        for fl in files[classes[clNum]]:
            if not os.path.exists(folder + '/' + featName + '/' + classes[clNum] + '/' + fl):
                continue
            FV = np.load(folder + '/' + featName + '/' + classes[clNum] + '/' + fl, allow_pickle=True)
            FV = get_feature_patches(FV, patch_size, patch_shift, input_shape)
            if np.size(data)<=1:
                data = FV
            else:
                # print('data: ', np.shape(data), np.shape(FV))
                data = np.append(data, FV, 0)
            label.extend([clNum]*np.shape(FV)[0])
    label = np.array(label, ndmin=2).T
    return data, label




# def getPerformance(PtdLabels, GroundTruths):
#     # ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels).flatten()
#     ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels)
#     Report = classification_report(y_true=GroundTruths, y_pred=PtdLabels, output_dict=True, labels=np.unique(GroundTruths))
#     fscore = []
#     Cl = np.unique(GroundTruths)
#     Report_keys = [key for key in Report.keys()]
#     for i in Cl:
#         fscore.append(np.round(Report[Report_keys[int(i)]]['f1-score'], 4))
#     mean_fscore = np.round(np.mean(fscore), 4)
#     fscore.append(mean_fscore)

#     return ConfMat, fscore


def getPerformance(PtdLabels, GroundTruths):
    ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels)
    fscore = f1_score(GroundTruths, PtdLabels, average=None, labels=[0,1]).tolist()
    mean_fscore = np.round(np.mean(fscore), 4)
    print('mean_fscore: ', mean_fscore)
    fscore.append(mean_fscore)

    return ConfMat, fscore



def print_results(PARAMS, folder, fName_suffix, **kwargs):
    if fName_suffix!='':
        opFile = folder + '/' + 'Performance_' + fName_suffix + '.csv'
    else:
        opFile = folder + '/' + 'Performance.csv'
        

    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = 'fold'
    values = str(PARAMS['fold'])
    for i in range(len(kwargs)):
        heading = heading + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[0]
        values = values + '\t' + np.squeeze(kwargs[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




def print_configuration(PARAMS):
    opFile = PARAMS['opDir'] + '/Configuration.csv'
    fid = open(opFile, 'a+', encoding = 'utf-8')
    PARAM_keys = [key for key in PARAMS.keys()]
    for i in range(len(PARAM_keys)):
        if PARAM_keys[i]=='GPU_session':
            continue
        # print('PARAMS key: ', PARAM_keys[i])
        try:
            fid.write(PARAM_keys[i] + '\t')
            fid.write(json.dumps(PARAMS[PARAM_keys[i]]))
            fid.write('\n')
        except:
            fid.write(PARAM_keys[i] + '\tERROR\n')
            
    fid.close()
    


def print_analysis(opFile, results):
    linecount = 0
    if os.path.exists(opFile):
        with open(opFile, 'r', encoding='utf8') as fid:
            for line in fid:
                linecount += 1    
    
    fid = open(opFile, 'a+', encoding = 'utf-8')
    heading = ''
    values = ''
    for i in range(len(results)):
        if heading=='':
            heading = np.squeeze(results[str(i)]).tolist().split(':')[0]
            values = np.squeeze(results[str(i)]).tolist().split(':')[1]
        else:
            heading += '\t' + np.squeeze(results[str(i)]).tolist().split(':')[0]
            values += '\t' + np.squeeze(results[str(i)]).tolist().split(':')[1]

    if linecount==0:
        fid.write(heading + '\n' + values + '\n')
    else:
        fid.write(values + '\n')
        
    fid.close()




def get_feature_patches(FV, patch_size, patch_shift, input_shape):
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
    if np.shape(patches_mean_var)[1]!=2*input_shape[0]: # This condition checks for 39CC
        if np.shape(patches_mean_var)[1]==44:
            patches_mean_var = patches_mean_var[:,list(range(0,21))+list(range(22,43))]
        elif np.shape(patches_mean_var)[1]==78:
            first_7_cep_dim = np.array(list(range(0,7))+list(range(13,20))+list(range(26,33))+list(range(39,46))+list(range(52,59))+list(range(65,72)))
            patches_mean_var = patches_mean_var[:, first_7_cep_dim]
    # print('patches_mean_var: ', np.shape(patches_mean_var))
    
    return patches_mean_var




def textgrids2csv(path):
    path += '/'
    root, dirs, all_files = next(os.walk(path))
    textgrid_idx = np.squeeze(np.where([fl.split('.')[-1]=='TextGrid' for fl in all_files]))
    print(textgrid_idx)
    files = [all_files[idx] for idx in textgrid_idx]
    print(files)
    
    for fl in files:
        grid = textgrids.TextGrid(path+fl)
        clip = grid['clip']
        print(clip)

        opFile = path + '/' + fl.split('.')[0] + '.csv'
        fid = open(opFile, 'a+', encoding = 'utf-8')
        heading = 'label,duration,xmin,xmax'
        fid.write(heading + '\n')
        for intval in clip:
            print(intval.text, intval.dur, intval.xmin, intval.xmax)
            values = intval.text + ',' + str(intval.dur) + ',' + str(intval.xmin) + ',' + str(intval.xmax)
            fid.write(values + '\n')    
        fid.close()
