import random as rnd


def shuffle(training_data, training_labels):
    shuffled_data = []
    shuffled_labels = []

    size = len(training_data)
    random_indices = rnd.sample(range(0, size), size)

    for i in range(0, len(random_indices)):
        shuffled_data.append(training_data[random_indices[i]])
        shuffled_labels.append(training_labels[random_indices[i]])

    return shuffled_data, shuffled_labels


def binarize_labels(labels):
    setosa = [1, 0, 0]
    verginicia = [0, 1, 0]
    versicolor = [0, 0, 1]

    new_labels = []

    for i in range(0, len(labels)):
        if labels[i] == "Iris-versicolor":
            new_labels.append(versicolor)
        elif labels[i] == "Iris-virginica":
            new_labels.append(verginicia)
        else:
            new_labels.append(setosa)

    return new_labels


