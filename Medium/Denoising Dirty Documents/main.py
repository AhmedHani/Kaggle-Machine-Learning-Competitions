__author__ = 'Ahmed Hani Ibrahim'
import PIL
import numpy
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from load_train_data import load_training_images
from load_test_data import load_testing_images

training_images_path = "G:\\Github Repositories\\KaggleMachineLearningCompetitions\\Medium\\Denoising Dirty Documents\\data\\train"
testing_images_path = "G:\\Github Repositories\\KaggleMachineLearningCompetitions\\Medium\\Denoising Dirty Documents\\data\\test"
training_images_collection = load_training_images(training_images_path)
testing_images_collection = load_testing_images(testing_images_path)




