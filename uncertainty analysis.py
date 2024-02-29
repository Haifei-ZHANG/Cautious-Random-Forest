# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:10:02 2024

@author: zhanghai
"""

import numpy as np
import pandas as pd
from math import floor, ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from CautiousRandomForest import CautiousRandomForest

if __name__ == "__main__":
    data_names = ['balance_scale', 'ecoli', 'forest', 'glass', 'letter', 'libras', 'optdigits', 'page_blocks',
                'seeds','spectrometer', 'vehicle', 'vowel', 'waveform', 'wine_quality', 'yeast']
    
    data_names = ['glass']
    n_bins = 20
    n_it = 5
    K = 5
    det_evaluations = np.zeros((len(data_names), n_bins))
    recall_evaluations = np.zeros((len(data_names), n_bins))
    acc_evaluations = np.zeros((len(data_names), n_bins))
    for d in range(len(data_names)):
        data_name = data_names[d]
        print(data_name)
        data = pd.read_csv("data/{}.csv".format(data_name))
        X = np.array(data.iloc[:,:-1])
        y = np.array(data.iloc[:,-1])
        
        for it in range(n_it):
            print('Iteration： ', it)
            kf = KFold(n_splits=K, shuffle=True)
            
            for train_index, test_index in tqdm(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
                probabilistic_accuracy = np.zeros(len(y_test))
                credal_accuracy = np.zeros(len(y_test))
                
                crf = CautiousRandomForest(n_trees=100, s=2, combination='cdm-vote', risk_threshold=0.5, random_state=None)
                crf.fit(X_train, y_train)
            
                predicted_probabilities = crf.rf.predict_proba(X_test)
                
                max_p_index = np.argsort(predicted_probabilities)[:, -1]
                second_max_p_index = np.argsort(predicted_probabilities)[:, -2]
                
                margins = np.zeros(len(y_test))
                for i in range(len(y_test)):
                    margins[i] = predicted_probabilities[i, max_p_index[i]] - predicted_probabilities[i, second_max_p_index[i]]
                
                instance_order = np.argsort(margins)
                
                step = floor(len(y_test)/n_bins)
                index_list = np.zeros(n_bins+1, 'int')
                index_list[1] = len(y_test) - (n_bins-1)*step
                for i in range(2, n_bins+1):
                    index_list[i] = index_list[i-1] + step
                
                for i in range(1, n_bins+1):
                    # index = instance_order[index_list[i-1]:index_list[i]]
                    index = instance_order[:index_list[i]]
                    eva = crf.score(X_test[index], y_test[index])
                    det_evaluations[d, i-1] += eva['determinacy']
                    if eva['single accuracy'] is None:
                        recall_evaluations[d, i-1] += eva['set accuracy']
                    elif eva['set accuracy'] is None:
                        recall_evaluations[d, i-1] += eva['single accuracy']
                    else:
                        recall_evaluations[d, i-1] += eva['determinacy']*eva['single accuracy'] + (1-eva['determinacy'])*eva['set accuracy']
                    acc_evaluations[d, i-1] += crf.rf.score(X_test[index], y_test[index])
        
        det_evaluations[d] /= (n_it*K)
        recall_evaluations[d] /= (n_it*K)
        recall_evaluations[d] = 1 - recall_evaluations[d]
        acc_evaluations[d] /= (n_it*K)
        acc_evaluations[d] = 1 - acc_evaluations[d]
        print(det_evaluations[d])
        print(recall_evaluations[d])
        print(acc_evaluations[d])
        
        
        fig = plt.figure(figsize=(6,4))

        # plt.bar(range(1, n_bins+1), det_evaluations[d]/(n_it*K))
        
        ax = fig.add_subplot(111)
        x_list = np.linspace(0.05, 1, 20).round(2)
        lin2 = ax.plot(x_list, recall_evaluations[d], linestyle='--', linewidth=2, color='tab:red', label = 'CDM risk')
        lin3 = ax.plot(x_list, acc_evaluations[d], linestyle='--', linewidth=2, color='tab:green', label = 'RF risk')
        ax2 = ax.twinx()
        lin1 = ax2.plot(x_list, det_evaluations[d], linestyle='-', linewidth=2, color='tab:blue',  label = 'Determinacy')
        
        
        
        # added these three lines
        lins = lin1+lin2+lin3
        labs = [l.get_label() for l in lins]
        ax.legend(lins, labs, loc='right')
        ax.grid()
        ax.set_xlabel("quantile of margin", fontsize=12)
        ax.set_ylabel("risk", fontsize=12)
        ax2.set_ylabel("determinacy", color='tab:blue', fontsize=12)
        for tl in ax2.get_yticklabels():
            tl.set_color("tab:blue")
        
        plt.savefig('results/uncertainty analysis/{}.png'.format(data_names[d]), bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        