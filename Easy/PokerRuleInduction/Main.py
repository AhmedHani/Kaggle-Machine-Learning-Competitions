__author__ = 'Ahmed Hani Ibrahim'

import pandas as pnd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

#Problem link: https://www.kaggle.com/c/poker-rule-induction

def getTrainingData():
    print("Get training data ...\n")

    trainingData = pnd.read_csv("./train.csv")
    trainingData['id'] = range(1, len(trainingData) + 1)

    return trainingData

def getTestData():
    print("Get testing data ...\n")

    testData = pnd.read_csv("./test.csv")
    result = pnd.DataFrame(testData['id'])
    testData = testData.drop(['id'], axis=1)

    return testData, result

def kFoldCrossValidation(kFold):
    trainingData = getTrainingData()
    label = trainingData['hand']
    features = trainingData.drop(['id'], axis=1)
    crossValidationResult = dict()

    print("Start Cross Validation ...\n")

    randomForest = RandomForestClassifier(n_estimators=100)
    kNearestNeighbour = KNeighborsClassifier(n_neighbors=100)
    crossValidationResult['RF'] = cross_val_score(randomForest, trainingData, label, cv=kFold).mean()
    crossValidationResult['KNN'] = cross_val_score(kNearestNeighbour, trainingData, label, cv=kFold).mean()

    print("KNN: %s\n" % str(crossValidationResult['KNN']))
    print("RF: %s\n" % str(crossValidationResult['RF']))
    print("\n")

    return crossValidationResult['KNN'], crossValidationResult['RF']

if __name__ == '__main__':
    trainingData = getTrainingData()
    labels = trainingData['hand']
    features = trainingData.drop(['id', 'hand'], axis=1)

    KNN, RF = kFoldCrossValidation(5)
    classifier = None

    if KNN > RF:
        classifier = KNeighborsClassifier(n_neighbors=100)
    else:
        classifier = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    testData, result = getTestData()

    print("Classification in progress ...\n")

    classifier.fit(features, labels)
    result.insert(1, 'hand', classifier.predict(testData))
    result.to_csv("./results.csv", index=False)

    print("Classification Ends ...\n")

