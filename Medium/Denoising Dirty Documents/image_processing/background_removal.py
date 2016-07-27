__author__ = 'Ahmed Hani Ibrahim'

from PIL import Image
import numpy
from scipy import signal
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from load_train_data import load_training_images
from load_test_data import load_testing_images
from save_testing_images import save_images
import PATH

KERNEL_SIZE = 15

training_images_path = PATH.TRAINING_DATA_DIR
testing_images_path = PATH.TESTING_DATA_DIR
training_images_collection = load_training_images(training_images_path)
testing_images_collection, images_id = load_testing_images(testing_images_path)

output_images = []
testing_cleaned_path = PATH.TESTING_RESULTS_IMAGE_PROCESSING

for i in range(0, len(testing_images_collection)):
    print(i)
    image = testing_images_collection[i]
    #plt.imshow(image)
    #plt.show()
    background = signal.medfilt2d(image, KERNEL_SIZE)
    #plt.imshow(background)
    #plt.show()
    #temp = image < background
    mask_text = image < background - 0.1
    new_image = numpy.where(mask_text, image, 1.0)
   #plt.imshow(new_image)
    #plt.show()
    output_images.append(new_image)

save_images(testing_cleaned_path, output_images, images_id)


