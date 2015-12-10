__author__ = 'Ahmed Hani Ibrahim'

import json
import scipy as sc
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

    return data


def encode_data(data):
    labels = [item['cuisine'] for item in data]
    unique_labels = set(labels)
    labels_dictionary = {}
    count = 0

    for label in unique_labels:
        labels_dictionary[label] = count
        count += 1

    ingredients = [item['ingredients'] for item in data]
    unique_ingredients = set(inner_item for outer_item in ingredients for inner_item in outer_item)
    ingredients_dictionary = {}
    count = 0

    for ingredient in unique_ingredients:
        ingredients_dictionary[ingredient] = count
        count += 1

    return labels, labels_dictionary, ingredients, ingredients_dictionary, data

def vectorize_data(labels, labels_dictionary, ingredients, ingredients_dictionary, data):
    labels_list = []
    ingredients_list = []

    for item in data:
        if u'cuisine' in item :
            label = str(item[u'cuisine'])
            if label in labels_dictionary:
                labels_list.append(labels_dictionary[label])
        if u'ingredients' in item:
            temp_ingredients = item[u'ingredients']
            temp_numerical_ingredients = []
            for ingredient in temp_ingredients:
                if ingredient in ingredients_dictionary:
                    index = ingredients_dictionary[ingredient]
                    temp_numerical_ingredients.append(index)
            ingredients_list.append(temp_numerical_ingredients)

    print(len(ingredients_list), len(labels_list))
    return (np.array(ingredients_list), np.array(labels_list))

#labels, labels_dictionary, ingredients, ingredients_dictionary, data = encode_data(get_train_data())
#features, classes = vectorize_data(labels, labels_dictionary, ingredients, ingredients_dictionary, data)





