import numpy as np
from itertools import chain
from collections import Counter
import copy as cp

"""
Vectorization based on this paper
http://www.aclweb.org/anthology/P15-2081
"""

class Vectorizer(object):
    def __init__(self, sentences):
        self.__sentences = sentences
        self.__char2onehot = {}
        self.__idx2char = {}

        self.__get_chars_vector()

    def __get_chars_vector(self):
        sentences_tokens = list(map(lambda v: v.split(), self.__sentences))
        sentences_tokens = list(chain.from_iterable(sentences_tokens))
        characters_frequencies = list(map(lambda v: Counter(v), sentences_tokens))
        all_counter = characters_frequencies[0]

        for i in range(1, len(characters_frequencies)):
            all_counter += characters_frequencies[i]

        one_hot = np.asarray([0] * len(all_counter.keys()))

        for i, char in enumerate(all_counter.keys()):
            temp_one_hot = cp.copy(one_hot)
            temp_one_hot[i] = 1

            self.__char2onehot[char] = temp_one_hot
            self.__idx2char[i] = char

    def get_char2onehot(self):
        return self.__char2onehot

    def get_char_vector(self, char):
        return self.__char2onehot[char]

    def text_to_vec(self, text, alpha=0.3):
        text_tokens = text.strip().split()
        z = np.asarray([0] * len(self.__char2onehot))

        for word in text_tokens:
            for char in word:
                z = np.add(np.multiply(z, alpha), self.__char2onehot[char] if char in self.__char2onehot else [0]*len(self.__char2onehot))

        return z

