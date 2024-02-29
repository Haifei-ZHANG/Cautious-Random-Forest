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
    data_names = ['balance_scale', 'ecoli', 'forest', 'glass', 'letter', 'libras', 'optdigits', 'page_blocks',
                'seeds','spectrometer', 'vehicle', 'vowel', 'waveform', 'wine_quality', 'yeast']
    
    data_names = ['vehicle', 'vowel', 'waveform', 'wine_quality', 'yeast']
    data_names = ['spectrometer']
    criteria = ['det', 'ssa']
    combinations = ['ave', 'mva', 'cdm-ave', 'cdm-vote']
    s_list = [1, 2, 3, 4, 5]
    n_it = 1
    K = 10
    n_tree = 50

    for d in range(len(data_names)):
        data_name = data_names[d]
        print("Data:", data_name)
        data = pd.read_csv('data/{}.csv'.format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])

        
        evaluation_for_data = np.zeros((n_it*K, len(criteria), len(s_list), len(combinations)))
        for it in range(n_it):
            print('Iteration： ', it)
            kf = KFold(n_splits=K, shuffle=True)
            j = 0
            for train_index, test_index in tqdm(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                for s in range(len(s_list)):
                    crf = CautiousRandomForest(n_trees=n_tree, s=s_list[s], combination='ave', risk_threshold=0.5, random_state=None)
                    crf.fit(X_train, y_train)
                    
                    for c in range(len(combinations)):
                        crf.combination = combinations[c]
                        eva = crf.score(X_test, y_test)
                        evaluation_for_data[it*K+j, 0, s, c] =  eva['determinacy']
                        evaluation_for_data[it*K+j, 1, s, c] =  eva['set size']
                j += 1


        evaluation_for_data_mean = np.zeros((len(criteria), len(s_list), len(combinations)))
        evaluation_for_data_std = np.zeros((len(criteria), len(s_list), len(combinations)))
        for crit in range(len(criteria)):
            for s in range(len(s_list)):
                for comb in range(len(combinations)):
                    valid_index = ~np.isnan(evaluation_for_data[:, crit,s,comb])
                    if crit == 1:
                        weights = 1 - evaluation_for_data[valid_index, 0, s, comb]
                        weights /= weights.sum()
                    else:
                        weights = np.ones(sum(valid_index))/sum(valid_index)
                    # weights = np.ones(sum(valid_index))/sum(valid_index)
                    valid_eva = evaluation_for_data[valid_index, crit, s, comb]
                    evaluation_for_data_mean[crit, s, comb] = sum(valid_eva * weights)
                    evaluation_for_data_std[crit, s, comb] = np.std(valid_eva)
        np.save('results/about S/{}_mean.npy'.format(data_name), evaluation_for_data_mean)
        np.save('results/about S/{}_std.npy'.format(data_name), evaluation_for_data_std)
        print(evaluation_for_data_mean)
    #     total_evaluation_mean[:,d,:] = evaluation_for_data_mean
    #     total_evaluation_std = evaluation_for_data_std
        
    # np.save('results/{}_noise/total_evaluation_mean.npy'.format(str(noise_level)), total_evaluation_mean)
    # np.save('results/{}_noise/total_evaluation_std.npy'.format(str(noise_level)), total_evaluation_std)