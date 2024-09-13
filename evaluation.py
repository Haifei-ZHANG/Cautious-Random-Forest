# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:48:16 2024

@author: zhanghai
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from CautiousRandomForest import CautiousRandomForest


if __name__ == "__main__":
    data_names = ['balance_scale', 'dermatology', 'ecoli', 'forest', 'glass',
                'letter', 'libras', 'optdigits', 'page_blocks', 'pendigits',
                'seeds', 'segment', 'spectrometer', 'vehicle', 'vowel',
                'waveform', 'wine', 'wine_quality', 'yeast']
    
    # data_names = ['balance_scale']
    criteria = ['det', 'ssa', 'sa', 'ss', 'u65', 'f1', 'u80']
    combinations = ['cr','ndc', 'ave', 'mva', 'cdm-ave', 'cdm-vote']
    s_list= [0.5, 1, 1.5, 2, 2.5]
    risk_list = np.linspace(0.05, 0.5, 10)
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    it = 2
    K = 5
    n_tree = 100
    for noise_level in noise_levels:
        total_evaluation_mean = np.zeros((len(criteria), len(data_names), len(combinations)))
        total_evaluation_std = np.zeros((len(criteria), len(data_names), len(combinations)))
        for d in range(len(data_names)):
            data_name = data_names[d]
            print('Noise Level:', str(noise_level), "Data:", data_name)
            data = pd.read_csv('data/{}.csv'.format(data_name))
            X = np.array(data.iloc[:,:-1])
            y = np.array(data.iloc[:,-1])
            classes = np.unique(y)
            evaluation_for_data = np.zeros((len(criteria), it*K, len(combinations)))
            for i in range(it):
                print('Iteration： ', i)
                kf = KFold(n_splits=K, shuffle=True)
                j = 0
                for train_index, test_index in tqdm(kf.split(X)):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
    
                    # in X_train, choose certain proportion of instance to change its label
                    if noise_level > 0:
                        instance_select = np.random.choice(len(y_train),int(noise_level*len(y_train)),replace=False)
                        for instance_index in instance_select:
                            candidate_y = np.setdiff1d(classes, y_train[instance_index])
                            y_train[instance_index] = candidate_y[np.random.choice(len(candidate_y), 1)[0]]
    
                    crf = CautiousRandomForest(n_trees=n_tree, combination='ndc', risk_threshold=0.5, random_state=42)
                    for c in range(len(combinations)):
                        combination = combinations[c]
                        crf.combination = combination
    
                        if combination in ['ndc', 'mva']:
                            crf.fit(X_train, y_train)
                        else:
                            inner_kf = KFold(n_splits=K, shuffle=True)
    
                            if combination=='cr':
                                inner_u65_evaluations = np.zeros(len(risk_list))
                                inner_clf = CautiousRandomForest(n_trees=n_tree,combination=combination, risk_threshold=0.5, random_state=42)
                                for inner_train_index, inner_val_index in inner_kf.split(X_train):
                                    inner_X_train, inner_X_val = X_train[inner_train_index], X_train[inner_val_index]
                                    inner_y_train, inner_y_val = y_train[inner_train_index], y_train[inner_val_index]
                                    inner_clf.fit(inner_X_train, inner_y_train)
                                    for r in range(len(risk_list)):
                                        inner_clf.risk_threshold = risk_list[r]
                                        inner_u65_evaluations[r] += inner_clf.score(inner_X_val, inner_y_val)['u65 score']
                                crf.risk_threshold = risk_list[np.argmax(inner_u65_evaluations)]
                                crf.fit(X_train, y_train)
                            else:
                                inner_u65_evaluations = np.zeros(len(s_list))
                                inner_clf = CautiousRandomForest(n_trees=n_tree, combination=combination, random_state=42)
                                for inner_train_index, inner_val_index in inner_kf.split(X_train):
                                    inner_X_train, inner_X_val = X_train[inner_train_index], X_train[inner_val_index]
                                    inner_y_train, inner_y_val = y_train[inner_train_index], y_train[inner_val_index]
                                    for r in range(len(s_list)):
                                        inner_clf.s = s_list[r]
                                        inner_clf.fit(inner_X_train, inner_y_train)
                                        inner_u65_evaluations[r] += inner_clf.score(inner_X_val, inner_y_val)['u65 score']
                                crf.s = s_list[np.argmax(inner_u65_evaluations)]
                                crf.fit(X_train, y_train)
                                
                        eva = crf.score(X_test, y_test)
                        eva = np.array(list(eva.values()))
                        evaluation_for_data[:,i*K+j ,c] = eva
    
                    j += 1
    
            np.save('results/{}_noise/{}_evaluation.npy'.format(str(noise_level), data_name), evaluation_for_data)
            evaluation_for_data_mean = np.zeros((len(criteria), len(combinations)))
            evaluation_for_data_std = np.zeros((len(criteria), len(combinations)))
            for crit in range(len(criteria)):
                for comb in range(len(combinations)):
                    valid_index = ~np.isnan(evaluation_for_data[crit,:,comb])
                    if crit in [2,3]:
                        weights = 1 - evaluation_for_data[0,valid_index,comb]
                        weights /= weights.sum()
                    elif crit==1:
                        weights = evaluation_for_data[0,valid_index,comb]
                        weights /= weights.sum()
                    else:
                        weights = np.ones(sum(valid_index))/sum(valid_index)
                    # weights = np.ones(sum(valid_index))/sum(valid_index)
                    valid_eva = evaluation_for_data[crit,valid_index,comb]
                    evaluation_for_data_mean[crit, comb] = sum(valid_eva * weights)
                    evaluation_for_data_std[crit, comb] = np.std(valid_eva)
            np.save('results/{}_noise/{}_mean.npy'.format(str(noise_level), data_name), evaluation_for_data_mean)
            np.save('results/{}_noise/{}_std.npy'.format(str(noise_level), data_name), evaluation_for_data_std)
            print(evaluation_for_data_mean)
            total_evaluation_mean[:,d,:] = evaluation_for_data_mean
            total_evaluation_std[:,d,:] = evaluation_for_data_std
            
        np.save('results/{}_noise/total_evaluation_mean.npy'.format(str(noise_level)), total_evaluation_mean)
        np.save('results/{}_noise/total_evaluation_std.npy'.format(str(noise_level)), total_evaluation_std)