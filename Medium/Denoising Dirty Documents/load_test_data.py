__author__ = 'Ahmed Hani Ibrahim'

from PIL import Image
import numpy as np
import scipy as sc
import os.path

def load_testing_images(images_path):
    images_id = []
    images_collection = []

    for i in range(1, 215):
        if i == 217:
            break

        image = images_path + "\\" + str(i) + ".png"
        if not os.path.isfile(image):
            continue
        images_id.append(i)
        images_collection.append(np.asarray(Image.open(image)) / 255.0)

    return images_collection, images_id
