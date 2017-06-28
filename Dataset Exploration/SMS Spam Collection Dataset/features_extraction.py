from data_analysis import DataManager
from vectorizer import Vectorizer
import numpy as np
import pickle
from tempfile import TemporaryFile


dm = DataManager('./data/spam.csv')
dm.most_frequent_character_in_spam()
dm.most_frequent_character_in_legit()
dm.most_frequent_characters()
dm.average_text_length()

sentences, labels = dm.get_text(), dm.get_labels()
labels = list(map(lambda v: 0 if v == 'ham' else 1, labels))
vectorizer = Vectorizer(sentences)

sentences_features = []

for sentence in sentences:
    sentence_vector = vectorizer.text_to_vec(sentence, alpha=0.3)
    sentences_features.append(sentence_vector)

train_x, train_y = sentences_features[0:5000], labels[0:5000]
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

test_x, test_y = sentences_features[5000:], labels[5000:]
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)

np.savetxt('train_x.txt', train_x)
np.savetxt('train_y.txt', train_y)
np.savetxt('test_x.txt', test_x)
np.savetxt('test_y.txt', test_y)






