__author__ = 'Ahmed Hani Ibrahim'

import csv
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB


def evaluate(classifier, training_data, k_fold):
    print 'CV...'
    print str(cross_val_score(classifier, training_data[0::, 1::], training_data[0::, 0], cv=k_fold).mean())