__author__ = 'Ahmed Hani Ibrahim'
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def get_train_data():
    training_data = pnd.read_csv("./train.csv", header=0, parse_dates=['Dates'])
    #training_data = pnd.read_csv("./train.csv", header=0)
    return training_data


def get_test_data():
    testing_data = pnd.read_csv("./test.csv", header=0, parse_dates=['Dates'])
    #testing_data = pnd.read_csv("./test.csv", header=0)
    return testing_data


def vectorize_training_data(training_data):
    training_data['Year'] = training_data['Dates'].map(lambda y: y.year)
    training_data['Week'] = training_data['Dates'].map(lambda w: w.week)
    training_data['Hour'] = training_data['Dates'].map(lambda h: h.hour)

    categories = list(enumerate(sorted(np.unique(training_data['Category']))))
    descripts = list(enumerate(sorted(np.unique(training_data['Descript']))))
    day_of_weeks = list(enumerate(sorted(np.unique(training_data['DayOfWeek']))))
    pd_districts = list(enumerate(sorted(np.unique(training_data['PdDistrict']))))
    resolutions = list(enumerate(sorted(np.unique(training_data['Resolution']))))
    #addresses = list(enumerate(sorted(np.unique(training_data['Address']))))

    #set indices
    categories_values = {name: i for i, name in categories}
    descripts_values = {name: i for i, name in descripts}
    day_of_weeks_values = {name: i for i, name in day_of_weeks}
    pd_districts_values = {name: i for i, name in pd_districts}
    resolutions_values = {name: i for i, name in resolutions}
    #addresses_values = {name: i for i, name in addresses}

    training_data['Category'] = training_data['Category'].map(lambda c: categories_values[c]).astype(int)
    training_data['Descript'] = training_data['Descript'].map(lambda c: descripts_values[c]).astype(int)
    training_data['DayOfWeek'] = training_data['DayOfWeek'].map(lambda c: day_of_weeks_values[c]).astype(int)
    training_data['PdDistrict'] = training_data['PdDistrict'].map(lambda c: pd_districts_values[c]).astype(int)
    training_data['Resolution'] = training_data['Resolution'].map(lambda c: resolutions_values[c]).astype(int)
    training_data['X'] = training_data['X'].map(lambda x: "%.2f" % round(x, 2)).astype(float)
    training_data['Y'] = training_data['Y'].map(lambda y: "%.2f" % round(y, 2)).astype(float)

    return training_data


def vectorize_testing_data(testing_data):
    testing_data['Year'] = testing_data['Dates'].map(lambda y: y.year)
    testing_data['Week'] = testing_data['Dates'].map(lambda w: w.week)
    testing_data['Hour'] = testing_data['Dates'].map(lambda h: h.hour)

    day_of_weeks = list(enumerate(sorted(np.unique(testing_data['DayOfWeek']))))
    pd_districts = list(enumerate(sorted(np.unique(testing_data['PdDistrict']))))

    day_of_weeks_values = {name: i for i, name in day_of_weeks}
    pd_districts_values = {name: i for i, name in pd_districts}

    testing_data['DayOfWeek'] = testing_data['DayOfWeek'].map(lambda c: day_of_weeks_values[c]).astype(int)
    testing_data['PdDistrict'] = testing_data['PdDistrict'].map(lambda c: pd_districts_values[c]).astype(int)
    testing_data['X'] = testing_data['X'].map(lambda x: "%.2f" % round(x, 2)).astype(float)
    testing_data['Y'] = testing_data['Y'].map(lambda y: "%.2f" % round(y, 2)).astype(float)

    return testing_data