# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 12:09:22 2015

@author: user
"""

import pickle
import numpy as np

def create_labels(warning_range, separation_distance):
    if len(warning_range) != len(separation_distance):
        raise Exception('Number of warning ranges and separation distances do not match')
    labels = [1 if warning_range[i] >= separation_distance[i] else 0 for i in range(len(warning_range))]
    return labels
    
def train_test_indices(data_length, training_percent):
    train_size = int(round(training_percent*data_length))
    train_indices = np.random.choice(data_length, train_size, replace=False)
    test_indices = [i for i in range(data_length) if i not in train_indices]
    return train_indices, test_indices

def train_test_split(training_data, testing_data, ground_truths, training_percent, classification):
    train_indices, test_indices = train_test_indices(len(training_data), training_percent)
    training_subset = [training_data[i] for i in train_indices]
    testing_subset = [testing_data[i] for i in test_indices]
    gt_subset = [ground_truths[i] for i in test_indices]
    if classification:
        training_labels = create_labels([gt[5] for gt in training_subset], [gt[4] for gt in training_subset])
        testing_labels = create_labels([vd[5] for vd in testing_subset], [vd[4] for vd in testing_subset])
        gt_labels = create_labels([gt[5] for gt in gt_subset], [gt[4] for gt in gt_subset])
        training_labels = np.asarray(training_labels, dtype=np.int32)
        testing_labels = np.asarray(testing_labels, dtype=np.int32)
        gt_labels = np.asarray(gt_labels, dtype=np.int32)   
    else:
        training_labels = [gt[4:] for gt in training_subset]
        testing_labels = [vd[4:] for vd in testing_subset]
        gt_labels = [gt[4:] for gt in gt_subset]
    return training_subset, training_labels, testing_subset, testing_labels, gt_labels
    
def k_folds(training_data, testing_data, ground_truths, training_percent, k=5, classification=True):
    folds = []    
    for i in range(k):
        folds.append(train_test_split(training_data, testing_data, ground_truths, training_percent, classification))
    return folds
    
def filter_important(ground_truths, training_data, testing_data, important_range=200):
    important_indices = [i for i in range(len(ground_truths)) if ((ground_truths[i][4] - ground_truths[i][5]) < important_range)]
    gt_filtered = [ground_truths[i] for i in important_indices]
    training_filtered = [training_data[i] for i in important_indices]
    testing_filtered = [testing_data[i] for i in important_indices]
    return gt_filtered, training_filtered, testing_filtered
        
class ConfusionMatrix:
    '''
        Confusion matrix.
                                 Actual data
         pred                 Negative   Positive
     Negative (safe)               a       c
     Positive (threatening)        b       d
    '''
    @staticmethod
    def both_safe(pred, gt):
         return (pred == gt and pred == 0)
    @staticmethod
    def both_threat(pred, gt):
         return (pred == gt and pred == 1)
    @staticmethod
    def false_threat(pred, gt):
        return (pred == 1 and gt == 0)
    @staticmethod
    def false_safe(pred, gt):
        return (pred == 0 and gt == 1)
    @staticmethod
    def total(f, pred, gt):
        return float(sum([1 if f(pred[i], gt[i]) else 0 for i in range(len(pred))]))
    
    def __init__(self, pred, gt):            
        self.a = self.total(self.both_safe, pred, gt)
        self.b = self.total(self.false_threat, pred, gt)
        self.c = self.total(self.false_safe, pred, gt)
        self.d = self.total(self.both_threat, pred, gt)
    
    def accuracy(self):
        return (self.a + self.d) / (self.a + self.b + self.c + self.d)
    def precision(self):
        return self.d / (self.b + self.d)
    def tp(self):
        return self.d / (self.c + self.d)
    def fn(self):
        return self.c / (self.c + self.d)
    def tn(self):
        return self.a / (self.a + self.b)
    def fp(self):
        return self.b / (self.a + self.b)
    def update(self, pred, gt):
        self.a = self.a + self.total(self.both_safe, pred, gt)
        self.b = self.b + self.total(self.false_threat, pred, gt)
        self.c = self.c + self.total(self.false_safe, pred, gt)
        self.d = self.d + self.total(self.both_threat, pred, gt)

class Scores:
    def __init__(self, alg):
        self.alg = alg
        self.cm = None
        self.acc = 0.0
        self.prec = 0.0
        self.correct_warnings = 0.0
        self.missed_warnings = 0.0
        self.correct_safety = 0.0
        self.missed_safety = 0.0
        self.t_avg = 0.0

    def log_scores(self, predictions, ground_truths):
        if self.cm is None:
            self.cm = ConfusionMatrix(predictions, ground_truths)
            self.acc =self.cm.accuracy()
            self.prec = self.cm.precision()
            self.correct_warnings = self.cm.tp()
            self.missed_warnings = self.cm.fn()
            self.correct_safety = self.cm.tn()
            self.missed_safety = self.cm.fp()
        else:
            self.cm.update(predictions, ground_truths)
            self.acc =self.cm.accuracy()
            self.prec = self.cm.precision()
            self.correct_warnings = self.cm.tp()
            self.missed_warnings = self.cm.fn()
            self.correct_safety = self.cm.tn()
            self.missed_safety = self.cm.fp()
        

class Results:
    def __init__(self, ml, camp, per_train, per_test, train_percent, important_only, n):
        self.ml = ml
        self.camp = camp
        self.per_train = per_train
        self.per_test = per_test
        self.train_percent = train_percent
        self.important_only = important_only
        self.n = n
    
    def save(self, results_file):
        with open(results_file, 'w') as f:
            pickle.dump(self, f)
    