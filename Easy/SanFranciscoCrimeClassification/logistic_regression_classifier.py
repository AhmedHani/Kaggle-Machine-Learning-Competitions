__author__ = 'Ahmed Hani Ibrahim'

import csv
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB
from get_data import *
from set_data import *
from evaluate_classifier import *

training_data = vectorize_training_data(get_train_data())
training_data = training_data.drop(['Dates', 'Descript', 'Resolution', 'Address', 'Hour', 'Week'], axis=1)

testing_data = vectorize_testing_data(get_test_data())
testing_data = testing_data.drop(['Id', 'Dates', 'Address', 'Hour', 'Week'], axis=1)

training_data = training_data.values
testing_data = testing_data.values

print "Training ..."

lr = LogisticRegression()
lr = lr.fit(training_data[0::, 1::], training_data[0::, 0])

evaluate(lr, training_data, 5)

print "Predicting ..."

res = lr.predict_proba(testing_data).astype(float)
res = res.tolist()

set_data(res)


