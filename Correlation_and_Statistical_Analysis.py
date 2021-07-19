#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:51:11 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import numpy as np
import os
import datetime
import lib.misc as misc
import configparser
from sklearn.cross_decomposition import CCA
from itertools import combinations
from scipy.stats import spearmanr
from statsmodels.multivariate.manova import MANOVA
import scipy.stats
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import pandas as pd
# import pingouin as pg




def __init__():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    section = config['Correlation_and_Statistical_Analysis.py']
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': section['folder'],
            'test_path': section['test_path'],
            'CNN_patch_size': int(section['CNN_patch_size']),
            'CNN_patch_shift': int(section['CNN_patch_shift']),
            'CNN_patch_shift_test': int(section['CNN_patch_shift_test']),
            'save_flag': section.getboolean('save_flag'),
            'CV_folds': int(section['CV_folds']),
            'data_balancing': section.getboolean('data_balancing'),
            'scale_data': section.getboolean('scale_data'),
            'PCA_flag': section.getboolean('PCA_flag'),
            'data_generator': False,
            'classes':{0:'music', 1:'speech'},
            'fold':0,
            'all_featName': ['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'Melspectrogram', 'HNGDMFCC', 'MGDCC', 'IFCC'], #['Khonglah_et_al', 'Sell_et_al', 'MFCC-39', 'HNGDMFCC', 'MGDCC', 'IFCC', 'Melspectrogram']
            'featName':'',
            '39_dim_CC_feat': section.getboolean('39_dim_CC_feat'),            
            }

    return PARAMS



if __name__ == '__main__':
    os.system('clear')
    PARAMS = __init__()
    
    for foldNum in [0]: #range(PARAMS['CV_folds']):
        PARAMS['fold'] = foldNum
        CORR = np.zeros((len(PARAMS['all_featName']), len(PARAMS['all_featName'])))
        SPEARMAN_CORR = np.zeros((len(PARAMS['all_featName']), len(PARAMS['all_featName'])))

        if PARAMS['test_path']=='':
            PARAMS['test_folder'] = PARAMS['folder']
            PARAMS['opDir'] = PARAMS['test_folder'] + '/__RESULTS/Correlation_and_Statistical_Analysis_' + PARAMS['today'] + '/'
        else:
            PARAMS['opDir'] = PARAMS['test_folder'] + '/__RESULTS/Correlation_and_Statistical_Analysis_GEN_PERF_' + PARAMS['today'] + '/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
        misc.print_configuration(PARAMS)
        
        All_feature_data = {}
        for PARAMS['featName'] in PARAMS['all_featName']:
            if PARAMS['test_path']=='':
                cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
                cv_file_list_test = cv_file_list
            else:
                PARAMS['test_folder'] = PARAMS['test_path']
                cv_file_list = misc.create_CV_folds(PARAMS['folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
                cv_file_list_test = misc.create_CV_folds(PARAMS['test_folder'], PARAMS['featName'], PARAMS['classes'], PARAMS['CV_folds'])
            
            PARAMS['train_files'], PARAMS['test_files'] = misc.get_train_test_files(cv_file_list, cv_file_list_test, PARAMS['CV_folds'], foldNum)

            PARAMS['input_shape'] = (21, PARAMS['CNN_patch_size'], 1)            
            if PARAMS['39_dim_CC_feat']:
                if (PARAMS['featName']=='HNGDMFCC') or (PARAMS['featName']=='MGDCC') or (PARAMS['featName']=='IFCC') or (PARAMS['featName']=='MFCC-39'):
                    PARAMS['input_shape'] = (39, PARAMS['CNN_patch_size'], 1)
            
            train_data, train_label = misc.load_data_from_files(PARAMS['classes'], PARAMS['folder'], PARAMS['featName'], PARAMS['train_files'], PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
            
            test_data, test_label = misc.load_data_from_files(PARAMS['classes'], PARAMS['test_folder'], PARAMS['featName'], PARAMS['test_files'], PARAMS['CNN_patch_size'], PARAMS['CNN_patch_shift'], PARAMS['input_shape'])
            print('train_data: ', np.shape(train_data))
            if np.shape(train_data)[1]==39:
                if not PARAMS['39_dim_CC_feat']:
                    train_data = train_data[:, list(range(0,7))+list(range(13,20))+list(range(26, 33))]
                    test_data = test_data[:, list(range(0,7))+list(range(13,20))+list(range(26, 33))]
            elif np.shape(train_data)[1]==22:
                train_data = train_data[:, :21]
                test_data = test_data[:, :21]
            
            train_data, train_label, test_data = misc.preprocess_data(PARAMS, train_data, train_label, test_data)
            print(PARAMS['featName'], 'Data preprocessed and loaded: ', np.shape(train_data), np.shape(test_data))
            All_feature_data[PARAMS['featName']] = {
                'train_data': train_data, 
                'train_label': train_label, 
                'test_data': test_data,
                'test_label': test_label,
                }
            
        all_feat_names = [key for key in PARAMS['all_featName']]
        combs = combinations(all_feat_names, 2)
        for comb_i in combs:
            # print('comb_i: ', comb_i)
            idx1 = np.squeeze(np.where([comb_i[0]==name for name in all_feat_names]))
            idx2 = np.squeeze(np.where([comb_i[1]==name for name in all_feat_names]))
            
            feat1_train = All_feature_data[comb_i[0]]['train_data']
            feat1_train_label = All_feature_data[comb_i[0]]['train_label']
            feat2_train = All_feature_data[comb_i[1]]['train_data']
            feat2_train_label = All_feature_data[comb_i[1]]['train_label']
            feat1_test = All_feature_data[comb_i[0]]['test_data']
            feat1_test_label = All_feature_data[comb_i[0]]['test_label']
            feat2_test = All_feature_data[comb_i[1]]['test_data']
            feat2_test_label = All_feature_data[comb_i[1]]['test_label']
            
            print('Original Sizes: ', np.shape(feat1_train), np.shape(feat2_train), np.shape(feat1_test), np.shape(feat2_test))
            min_pts_train = np.min([np.shape(feat1_train)[0], np.shape(feat2_train)[0]])
            feat1_train = feat1_train[:min_pts_train, :]
            feat1_train_label = feat1_train_label[:min_pts_train]
            feat2_train = feat2_train[:min_pts_train, :]
            feat2_train_label = feat2_train_label[:min_pts_train]
            
            min_pts_test = np.min([np.shape(feat1_test)[0], np.shape(feat2_test)[0]])
            feat1_test = feat1_test[:min_pts_test, :]
            feat1_test_label = feat1_test_label[:min_pts_test]
            feat2_test = feat2_test[:min_pts_test, :]
            feat2_test_label = feat2_test_label[:min_pts_test]
            print('Size adjusted: ', np.shape(feat1_train), np.shape(feat2_train), np.shape(feat1_test), np.shape(feat2_test), '\n')
            
            feat1 = np.append(feat1_train, feat1_test, axis=0)
            feat1_label = np.append(feat1_train_label, feat1_test_label)
            feat2 = np.append(feat2_train, feat2_test, axis=0)
            feat2_label = np.append(feat2_train_label, feat2_test_label)

            n_comp = 1 # np.min([np.shape(feat1)[1], np.shape(feat2)[1]])
            cca = CCA(n_components=n_comp, scale=False)
            cca.fit(feat1, feat2)
            feat1_c, feat2_c = cca.transform(feat1, feat2)

            ps_corrcoef, ps_p_value = scipy.stats.pearsonr(feat1_c.flatten(), feat2_c.flatten())
            
            sp_corrcoeff, sp_pvalue = spearmanr(feat1_c, feat2_c)

            CORR[idx1, idx2] = ps_corrcoef
            SPEARMAN_CORR[idx1, idx2] = sp_corrcoeff
            print('Correlation (', comb_i[0], comb_i[1],'): CCA=', ps_corrcoef, ' Spearman=', sp_corrcoeff, sp_pvalue, '\n')
            
        print(CORR)
        count1 = 0
        for featName1 in PARAMS['all_featName']:
            results = {}
            results['0'] = 'fold:'+str(PARAMS['fold'])
            results['1'] = 'featName:'+featName1
            count2 = 0
            for featName2 in PARAMS['all_featName']:
                results[str(count2+2)] = featName2+':'+str(CORR[count1, count2])
                count2 +=1
            count1 += 1
            opFile = PARAMS['opDir'] + '/CCA_results.csv'
            misc.print_analysis(opFile, results)

        print(SPEARMAN_CORR)
        count1 = 0
        for featName1 in PARAMS['all_featName']:
            results = {}
            results['0'] = 'fold:'+str(PARAMS['fold'])
            results['1'] = 'featName:'+featName1
            count2 = 0
            for featName2 in PARAMS['all_featName']:
                results[str(count2+2)] = featName2+':'+str(SPEARMAN_CORR[count1, count2])
                count2 +=1
            count1 += 1
            opFile = PARAMS['opDir'] + '/Spearman_Correlation_results.csv'
            misc.print_analysis(opFile, results)


        all_feat_names = [key for key in PARAMS['all_featName']]
        opFile = PARAMS['opDir'] + '/MANOVA.csv'
        for feat_i in all_feat_names:
            feat_train = All_feature_data[feat_i]['train_data']
            feat_test = All_feature_data[feat_i]['test_data']
            feat_train_label = All_feature_data[feat_i]['train_label']
            print('MANOVA ', feat_i, np.shape(feat_train), np.shape(feat_train_label), np.shape(feat_test))

            PARAMS_temp = PARAMS.copy()
            print('feat_train: ', np.shape(feat_train))
            
            # endog~dependent variables, exog~independent variables
            try:
                moav = MANOVA(endog=feat_train, exog=feat_train_label)
                test_results = moav.mv_test()
            except:
                print(feat_i, ' Noise added')
                feat_train += np.random.rand(np.shape(feat_train)[0], np.shape(feat_train)[1])*1e-10
                moav = MANOVA(endog=feat_train, exog=feat_train_label)
                test_results = moav.mv_test()
                
            WL = test_results.results['x0']['stat']['Value']['Wilks\' lambda']
            PT = test_results.results['x0']['stat']['Value']['Pillai\'s trace']
            HLT = test_results.results['x0']['stat']['Value']['Hotelling-Lawley trace']
            RGR = test_results.results['x0']['stat']['Value']['Roy\'s greatest root']
            results = {}
            results['0'] = 'fold:'+str(PARAMS['fold'])
            results['1'] = 'featName:'+feat_i
            results['2'] = 'Wilks\' lambda Value:'+str(WL)
            results['3'] = 'Pillai\'s trace Value:'+str(PT)
            results['4'] = 'Hotelling-Lawley trace Value:'+str(HLT)
            results['5'] = 'Roy\'s greatest root Value:'+str(RGR)
            opFile = PARAMS['opDir'] + '/MANOVA_results.csv'
            misc.print_analysis(opFile, results)
            




        # all_feat_names = [key for key in PARAMS['all_featName']]
        # opFile = PARAMS['opDir'] + '/ANOVA.csv'
        # for feat_i in all_feat_names:
        #     feat_train = All_feature_data[feat_i]['train_data']
        #     feat_test = All_feature_data[feat_i]['test_data']
        #     feat_train_label = All_feature_data[feat_i]['train_label']
        #     print('ANOVA ', feat_i, np.shape(feat_train), np.shape(feat_train_label), np.shape(feat_test))

        #     PARAMS_temp = PARAMS.copy()
        #     feat_train, feat_train_label, feat_test = misc.preprocess_data(PARAMS_temp, feat_train, feat_train_label, feat_test)
            
        #     data = {}
        #     between = []
        #     rand_idx = np.random.randint(0,np.shape(feat_train)[0],100)
        #     for i in range(np.shape(feat_train)[1]):
        #         data['Column'+str(i)] = feat_train[rand_idx, i]
        #         between.append('Column'+str(i))
        #     data['Labels'] = feat_train_label[rand_idx,0].flatten()
        #     print('data', np.shape(data))
        #     print('between: ', between)
        #     dataset = pd.DataFrame(data)
        #     aov = pg.anova(dv='Labels', between=between, data=dataset, detailed=True)
        #     aov.round(3)
            
        #     # mod = ols('weight~group', data=feat_train).fit()
        #     # table = sm.stats.anova_lm(mod, typ=1)
        #     # print(table)

        #     # results = {}
        #     # results['0'] = 'fold:'+str(PARAMS['fold'])
        #     # results['1'] = 'featName:'+feat_i
        #     # results['2'] = 'Wilks\' lambda Value:'+str(WL)
        #     # results['3'] = 'Pillai\'s trace Value:'+str(PT)
        #     # results['4'] = 'Hotelling-Lawley trace Value:'+str(HLT)
        #     # results['5'] = 'Roy\'s greatest root Value:'+str(RGR)
        #     # opFile = PARAMS['opDir'] + '/MANOVA_results.csv'
        #     # misc.print_analysis(opFile, results)
            


