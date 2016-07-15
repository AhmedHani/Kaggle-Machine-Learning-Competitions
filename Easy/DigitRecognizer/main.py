__author__ = 'Ahmed Hani Ibrahim'

from read_data import *
import numpy as np
import pickle
from draw_data import *
from get_image import *
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn import svm

labels, train_features = read_train_data(
    "G:\\Github Repositories\\KaggleMachineLearningCompetitions\\Easy\\DigitRecognizer\\train.csv")
train_features = np.array(train_features).reshape((len(train_features), 28, 28)).astype(np.uint8)
print(len(train_features))

sample_image = get_image(train_features, 2)
image_black_pixels_x = pickle.load(open('G:\Github Repositories\KaggleMachineLearningCompetitions\Easy\DigitRecognizer\image_black_pixels_x', "rb"))
image_black_pixels_y = pickle.load(open('G:\Github Repositories\KaggleMachineLearningCompetitions\Easy\DigitRecognizer\image_black_pixels_y', "rb"))

'''index = 0

for t in range(len(train_features)):
    for i in range(len(train_features[t])):
        image_black_pixels_x.append([])
        image_black_pixels_y.append([])
        for j in range(len(train_features[i])):
            if train_features[t][i][j] > 0:
                image_black_pixels_x[t].append(i)
                image_black_pixels_y[t].append(j)'''

#print(len(image_black_pixels_x))

max_length = 0
for i in range(len(train_features)):
    if len(image_black_pixels_x[i]) > max_length:
        max_length = len(image_black_pixels_x[i])

for t in range(len(train_features)):
    if len(image_black_pixels_x[t]) < max_length:
        for i in range(max_length - len(image_black_pixels_x[t])):
            image_black_pixels_x[t].append(0)
            image_black_pixels_y[t].append(0)

features_2_data = []

for t in range(len(train_features)):
    features_2_data.append([])
    for i in range(len(image_black_pixels_x[t])):
        bx = image_black_pixels_x[t][i]
        by = image_black_pixels_y[t][i]
        features_2_data[t].append([bx, by])

# for i in range(len(sample_image)):
#    idx = 0
#   for j in range(len(sample_image[i])):
#      if sample_image[i][j] > 0:
#         v = sample_image[i][j]
#        image_black_pixels_x.append(i)
#       image_black_pixels_y.append(j)
#idx += 1

#show_image(sample_image)
#print(image_black_pixels_x)
#print(image_black_pixels_y)

print(len(features_2_data[2][1]))

'''svm_classifier =
svm_classifier.fit(features_2_data, labels)'''

lr = LogisticRegression()
lr.fit(features_2_data, labels)

