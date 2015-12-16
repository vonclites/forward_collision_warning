# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 15:20:00 2015

@author: user
"""

import os
import csv
import numpy as np
from sklearn import linear_model

vehicle_data_directory = r'./ml_data/v_data/'
ground_truth_directory = r'./ml_data/ground_truths/'

vehicle_data = []
for f in os.listdir(vehicle_data_directory):
    with open(vehicle_data_directory + f) as fx:
        reader = csv.reader(fx)
        for line in reader:
            vehicle_data.append(map(float, line))

ground_truths = []
for f in os.listdir(ground_truth_directory):
    with open(ground_truth_directory + f) as fx:
        reader = csv.reader(fx)
        for line in reader:
            ground_truths.append(map(float, line))

clf = linear_model.LinearRegression()

av_data = [vd[:4] for vd in vehicle_data]  #Accel / Vel for LV & FV
est_sep = [vd[4] for vd in vehicle_data] #Estimated Separation Distances
gt_wr = [gt[0] for gt in ground_truths] #Ground Truth Warning Ranges
gt_sep = [gt[1] for gt in ground_truths] #Ground Truth Separation Distances


split = int(round(.4 * len(vehicle_data)))

X_train = av_data[:split]
y_train = gt_wr[:split]

X_test = av_data[split:]
y_test = gt_wr[split:]

clf.fit(X_train, y_train)

pred_wranges = clf.predict(X_test)
pred_warnings = [1 if pred_wranges[i] >= est_sep[i] else 0 for i in range(0,len(pred_wranges))]
gt_warnings = [1 if ground_truth[0] >= ground_truth[1] else 0 for ground_truth in ground_truths]

correct = 0
for i in range(0, len(pred_warnings)):
    if pred_warnings[i] == gt_warnings[i]:
        correct = correct + 1

print ("Correct:", correct)        
print ("Total:",len(pred_warnings))
print('Accuracy: %.3f' % (float(correct) / float(len(pred_warnings))))


# The coefficients
print('Coefficients:', clf.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.3f' % clf.score(X_test, y_test))


