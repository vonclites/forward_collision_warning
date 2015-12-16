# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 15:20:00 2015

@author: user
"""
import time
import data_reader as dr
import data_utils as du
from sklearn import linear_model

total_files = 10

def run(clf, per_train = 0, per_test = 0, train_percent = 40, important_only = True, n = total_files):
    ground_truth_directory = r'./ml_data/v_data_per0/'
    training_directory = r'./ml_data/v_data_per' + str(per_train) + '/'
    testing_directory = r'./ml_data/v_data_per' + str(per_test) + '/'
    
    train = str(per_train)+'PER'
    results_directory = r'./ml_results/LR/'
    name = 'LR_' + train + str(train_percent) + '_Train_'+ str(per_test) + 'PER_Test_IO_' + str(important_only) 
    results_file = results_directory + name
    
    ground_truths = dr.read_vehicle_data(ground_truth_directory, n)
    training_data = dr.read_vehicle_data(training_directory, n)
    testing_data = dr.read_vehicle_data(testing_directory, n)

    if important_only:
        ground_truths, training_data, testing_data = du.filter_important(ground_truths, training_data, testing_data)
        n = len(ground_truths)
        
    t_avg = 0
    folds = du.k_folds(training_data, testing_data, ground_truths, float(train_percent)/100.0, classification=False)
    camp = du.Scores('CAMP')
    ml = du.Scores('LR')
    
    print "Beginning Training and Testing"
    clf = clf[1]
    for fold in folds:
        x_train, y_train, x_test, y_camp, y_gt = fold
        y_train = [y[1] for y in y_train]
        y_camp_labels = du.create_labels([l[1] for l in y_camp], [l[0] for l in y_camp])
        y_gt_labels = du.create_labels([l[1] for l in y_gt], [l[0] for l in y_gt])
        camp.log_scores(y_camp_labels, y_gt_labels)
        clf.fit([x[:5] for x in x_train], y_train)
        t0 = time.clock()
        predictions = clf.predict([x[:5] for x in x_test])
        t1 = time.clock()
        t_avg = t_avg + float(t1 - t0) / float(len(x_test))
        prediction_labels = du.create_labels(predictions, [x[4] for x in x_test])
        ml.log_scores(prediction_labels, y_gt_labels)

    ml.t_avg = t_avg / float(len(folds))
    print "Saving Results"
    results = du.Results(ml, camp, per_train, per_test, train_percent, important_only, n)
    results.save(results_file)
    
    print 'ML:'
    print ml.acc
    print ml.prec
    print 'Camp:'
    print camp.acc
    print camp.prec
    print '----------'

clf = ('LR', linear_model.LinearRegression())
print 'Evaluating ' + clf[0]
run (clf, per_train = 0, per_test = 0, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 10, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 30, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 60, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 90, train_percent = 40, important_only = True, n = total_files)
#Training on imperfect data
run (clf, per_train = 10, per_test = 10, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 30, per_test = 30, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 60, per_test = 60, train_percent = 40, important_only = True, n = total_files)
run (clf, per_train = 90, per_test = 90, train_percent = 40, important_only = True, n = total_files)
#Varying training size
run (clf, per_train = 0, per_test = 30, train_percent = 1, important_only = True, n = total_files)    
run (clf, per_train = 0, per_test = 30, train_percent = 5, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 30, train_percent = 10, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 30, train_percent = 20, important_only = True, n = total_files)
run (clf, per_train = 0, per_test = 30, train_percent = 30, important_only = True, n = total_files)    
run (clf, per_train = 0, per_test = 30, train_percent = 60, important_only = True, n = total_files)
