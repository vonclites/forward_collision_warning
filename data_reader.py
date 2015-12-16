# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 20:22:37 2015

@author: user
"""

import os
import csv

def read_vehicle_data(vehicle_data_directory, n=823):
    vehicle_data = []
    count = 1
    for f in os.listdir(vehicle_data_directory):
        if count > n:
            break
        with open(vehicle_data_directory + f) as fx:
            reader = csv.reader(fx)
            for line in reader:
                vehicle_data.append(map(float, line))
        count = count + 1
    return vehicle_data