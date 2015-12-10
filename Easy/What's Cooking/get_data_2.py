__author__ = 'Ahmed Hani Ibrahim'

import json
import scipy.sparse
import numpy as np


def get_train_data():
    with open('./train.json') as r:
        data = json.load(r)
        r.close()

    return data


def get_test_data():
    with open('./test.json') as r:
        data = json.load(r)
        r.close()

    ids = [item['id'] for item in data]

    return data, ids


def get_training_data_matrix(data):
    labels = [item['cuisine'] for item in data]
    unique_labels = set(labels)
    ingredients = [item['ingredients'] for item in data]
    unique_ingredients = set(inner_item for outer_item in ingredients for inner_item in outer_item)

    training_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

    for i, item in enumerate(ingredients):
        for j, ing in enumerate(unique_ingredients):
            if ing in item:
                training_data_matrix[i, j] = 1

    return labels, training_data_matrix, unique_ingredients


def get_test_data_matrix(data, unique_ingredients):
    ingredients = [item['ingredients'] for item in data]
    test_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

    for i, item in enumerate(ingredients):
        for j, ing in enumerate(unique_ingredients):
            if ing in item:
                test_data_matrix[i, j] = 1

    return test_data_matrix

