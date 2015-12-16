# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 11:58:13 2015

@author: user
"""
import time
import data_reader as dr
import data_utils as du
from sklearn import svm
from sklearn import linear_model
from sklearn import tree

#total timesteps = 364531
total_files = 823

def run(clf, per_train = 0, per_test = 0, train_percent = 40, important_only = True, n = total_files):
    clf_name = clf[0]
    clf = clf[1]
    train = str(per_train)+'PER'
    results_directory = r'./ml_results/' + clf_name + '/'
    name = clf_name + '_' + train + str(train_percent) + '_Train_'+ str(per_test) + 'PER_Test_IO_' + str(important_only) 
    results_file = results_directory + name
    
    ground_truth_directory = r'./ml_data/v_data_per0/'
    training_directory = r'./ml_data/v_data_per' + str(per_train) + '/'
    testing_directory = r'./ml_data/v_data_per' + str(per_test) + '/'
    
    ground_truths = dr.read_vehicle_data(ground_truth_directory, n)
    training_data = dr.read_vehicle_data(training_directory, n)
    testing_data = dr.read_vehicle_data(testing_directory, n)
    
    if important_only:
        ground_truths, training_data, testing_data = du.filter_important(ground_truths, training_data, testing_data)
        n = len(ground_truths)
    
    t_avg = 0
    folds = du.k_folds(training_data, testing_data, ground_truths, float(train_percent)/100.0, classification=True)
    camp = du.Scores('CAMP')
    ml = du.Scores(clf_name)
    print "Beginning Training and Testing"
    for fold in folds:
        x_train, y_train, x_test, y_camp, y_gt = fold
        camp.log_scores(y_camp, y_gt)
        clf.fit([x[:5] for x in x_train], y_train)
        t0 = time.clock()
        predictions = clf.predict([x[:5] for x in x_test])
        t1 = time.clock()
        t_avg = t_avg + float(t1 - t0) / float(len(x_test))
        ml.log_scores(predictions, y_gt)
        
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
    
def evaluate_classifier(clf):
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
    
if __name__ == "__main__":
   #evaluate_classifier(('SGD', linear_model.SGDClassifier(shuffle=True)))
   '''
   clf1 = ('SVM', svm.SVC())
   clf2 = ('DT', tree.DecisionTreeClassifier())
   clf3 = ('SGD', linear_model.SGDClassifier())
   clfs = [clf1, clf2, clf3]
   map(evaluate_classifier, clfs)
   '''
   
   run (('DT', tree.DecisionTreeClassifier()), per_train = 0, per_test = 30, train_percent = 30, important_only = True, n = total_files)  
   