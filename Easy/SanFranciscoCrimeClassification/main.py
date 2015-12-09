__author__ = 'Ahmed Hani Ibrahim'

import csv
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB

from get_data import *
training_data = vectorize_training_data(get_train_data())
training_data = training_data.drop(['Dates', 'Descript', 'Resolution', 'Address', 'Hour', 'Week'], axis=1)

testing_data = vectorize_testing_data(get_test_data())
testing_data = testing_data.drop(['Id', 'Dates', 'Address', 'Hour', 'Week'], axis=1)

training_data = training_data.values
testing_data = testing_data.values



