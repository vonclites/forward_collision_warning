# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:24:45 2015

@author: user
"""

import os
import pickle
import matplotlib.pyplot as plt

svm_directory = r'./ml_results/SVM/'
sgd_directory = r'./ml_results/SGD/'
dt_directory = r'./ml_results/DT/'
lr_directory = r'./ml_results/LR/'

def read_results(directory):
    results = []
    for f in os.listdir(directory):
        r = pickle.load(file(directory+f))
        results.append(r)
    return results
    
svm_results = read_results(svm_directory)
sgd_results = read_results(sgd_directory)
dt_results = read_results(dt_directory)
lr_results = read_results(lr_directory)

def per_results(results):
    per_trials = []
    for result in results:
        if result.per_train == 0 and result.train_percent == 40:
            per_trials.append(result)
    return per_trials
            
def train_results(results):
    train_trials = []
    for result in results:
        if result.per_train == 0 and result.per_test == 30:
            train_trials.append(result)
    return sorted(train_trials, key=lambda r: r.train_percent)
    
def network_results(results):
    network_trials = []
    for result in results:
        if result.per_train == result.per_test:
            network_trials.append(result)
    return network_trials
    
variable_test_per = {}
variable_test_per ['SVM'] = per_results(svm_results)
variable_test_per ['SGD'] = per_results(sgd_results)
variable_test_per ['DT'] = per_results(dt_results)
variable_test_per ['LR'] = per_results(lr_results)

variable_train_per = {}
variable_train_per['SVM'] = network_results(svm_results)
variable_train_per['SGD'] = network_results(sgd_results)
variable_train_per['DT'] = network_results(dt_results)
variable_train_per['LR'] = network_results(lr_results)

variable_train_size = {}
variable_train_size['SVM'] = train_results(svm_results)
variable_train_size['SGD'] = train_results(sgd_results)
variable_train_size['DT'] = train_results(dt_results)
variable_train_size['LR'] = train_results(lr_results) 

def grapher(results, x_lim, x_label, x_getter, y_lim, y_label, y_getter, legend_loc, f):
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(x_lim[0],x_lim[1])
    plt.ylim(y_lim[0],y_lim[1])
    plt.plot([x_getter(results['SVM'][i]) for i in range(len(results['SVM']))], [y_getter(results['SVM'][i].ml) for i in range(len(results['SVM']))], label = "SVM", marker='8')
    plt.plot([x_getter(results['SGD'][i]) for i in range(len(results['SGD']))], [y_getter(results['SGD'][i].ml) for i in range(len(results['SGD']))], label = "SGD", marker="d")
    plt.plot([x_getter(results['DT'][i]) for i in range(len(results['DT']))], [y_getter(results['DT'][i].ml) for i in range(len(results['DT']))], label = "DT", marker=">")
    plt.plot([x_getter(results['LR'][i]) for i in range(len(results['LR']))], [y_getter(results['LR'][i].ml) for i in range(len(results['LR']))], label = "LR", marker="v")
    plt.plot([x_getter(results['SVM'][i]) for i in range(len(results['SVM']))], [y_getter(results['SVM'][i].camp) for i in range(len(results['SVM']))], label = "CAMP", marker="s")
    plt.legend(loc=legend_loc)
    plt.savefig(f)
    plt.show()

grapher(variable_test_per, (0,90), "Testing PER", lambda r: r.per_test, (0.6, 1.0), "Accuracy", lambda s: s.acc, 3, r'./ml_results/vary_test_per_acc.png')
grapher(variable_train_per, (0,90), "Training & Testing PER", lambda r: r.per_train, (0.6, 1.0), "Accuracy", lambda s: s.acc, 3, r'./ml_results/vary_train_per_acc.png')
grapher(variable_train_size, (0,60), "Training Size Percent", lambda r: r.train_percent, (0.6, 1.0), "Accuracy", lambda s: s.acc, 3, r'./ml_results/vary_train_size_acc.png')

grapher(variable_test_per, (0,90), "Testing PER", lambda r: r.per_test, (0.0, 1.0), "Precision", lambda s: s.prec, 3, r'./ml_results/vary_test_per_prec.png')
grapher(variable_train_per, (0,90), "Training & Testing PER", lambda r: r.per_train, (0.0, 1.0), "Precision", lambda s: s.prec, 3, r'./ml_results/vary_train_per_prec.png')

grapher(variable_test_per, (0,90), "Testing PER", lambda r: r.per_test, (0.0, 1.0), "FNR", lambda s: s.missed_warnings, 4, r'./ml_results/vary_test_per_fnr.png')
grapher(variable_test_per, (0,90), "Testing PER", lambda r: r.per_test, (0.0, 0.3), "FPR", lambda s: s.missed_safety, 1, r'./ml_results/vary_test_per_fpr.png')

def table(results, y_getter):
    t = []
    header = [' ', '0%', '10%', '30%', '60%', '90%']
    #header = [' ', '1%', '5%', '10%', '20%', '30%', '40%', '60%']
    t.append(header)
    row1 = ['CAMP']
    for score in results['SVM']:
        row1.append(round(y_getter(score.camp), 3))
    t.append(row1)
    for result in results.iteritems():
        row = [result[0]]
        for i in range(len(result[1])):
            row.append(round(y_getter(result[1][i].ml), 3))
        t.append(row)
    return t

acc = table(variable_train_per, lambda s: s.correct_safety)


for row in acc:
    print ','.join(map(str,row))










