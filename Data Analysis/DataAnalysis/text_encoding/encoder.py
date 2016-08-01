__author__ = 'Ahmed Hani Ibrahim'


class TextEncoder(object):
    @classmethod
    def encode(cls, categories):
        categories_matrix = [[0 for i in range(0, len(categories))] for j in range(0, len(categories))]

        true_index = 0

        for i in range(0, len(categories)):
            categories[i][true_index] = categories[i]
            true_index += 1

        return categories_matrix

