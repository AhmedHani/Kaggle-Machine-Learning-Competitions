from data_reader.reader import CsvReader
from util import *

import tensorflow as tf
import matplotlib.pyplot as plt
import math

reader = CsvReader("/home/ahani/PycharmProjects/IrisClassification/data/Iris.csv")

iris_features, iris_labels = reader.get_iris_data()
iris_features, iris_labels = shuffle(iris_features, iris_labels)
binarized_labels = binarize_labels(iris_labels)

INPUT_NEURONS = 4
HIDDEN_NEURONS = 200
OUTPUT_NEURONS = 3

NUM_OF_EPOCHS = 1000

x = tf.placeholder(tf.float32, [None, INPUT_NEURONS])
y_target = tf.placeholder(tf.float32, [None, OUTPUT_NEURONS])

input_hidden_weights = tf.Variable(
                tf.random_uniform([INPUT_NEURONS, HIDDEN_NEURONS], -1.0 / math.sqrt(INPUT_NEURONS), 1.0 / math.sqrt(INPUT_NEURONS))) #Ini from the given network
input_hidden_bias = tf.Variable(tf.ones([HIDDEN_NEURONS])) # The bias in one
hidden_neurons_values = tf.matmul(x, input_hidden_weights) + input_hidden_bias
hidden_activation_result = tf.nn.softmax(hidden_neurons_values)

hidden_output_weights = tf.Variable(
                tf.random_uniform([HIDDEN_NEURONS, OUTPUT_NEURONS], -1.0 / math.sqrt(HIDDEN_NEURONS), 1.0 / math.sqrt(HIDDEN_NEURONS)))
hidden_output_bias = tf.Variable(tf.ones([OUTPUT_NEURONS])) # The bias in one
hidden_output_value = tf.matmul(hidden_activation_result, hidden_output_weights) + hidden_output_bias

y_estimated = tf.nn.softmax(hidden_output_value)

cross_entropy = -tf.reduce_mean((y_target * tf.log(y_estimated)) + ((1 - y_target) * tf.log(1 - y_estimated)))
mean_squared_error = 0.5 * tf.reduce_sum((tf.square(y_estimated - y_target)))

train = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

x_training_data = iris_features
y_training_labels = binarized_labels

errors = []
epochs = []

for i in range(0, NUM_OF_EPOCHS):
    session.run(train, feed_dict={x: x_training_data[0:100], y_target: y_training_labels[0:100]})

    if i % 10 == 0:
        error = session.run(cross_entropy, feed_dict={x: x_training_data[0:100], y_target: y_training_labels[0:100]}) #Change cross_entropy to mean_squared error
        print error
        print "Iteration", i
        errors.append(error)
        epochs.append(i)

        if error <= 0.2:
            print "Input to hidden Weights", session.run(input_hidden_weights, feed_dict={x: x_training_data[0:100], y_target: y_training_labels[0:100]}), "\n"
            print "Input to hidden bias", session.run(input_hidden_bias, feed_dict={x: x_training_data[0:100], y_target: y_training_labels[0:100]}), "\n"
            print "Hidden to output weights", session.run(hidden_output_weights, feed_dict={x: x_training_data[0:100], y_target: y_training_labels[0:100]}), "\n"
            print "Hidden to output bias", session.run(hidden_output_bias, feed_dict={x: x_training_data[0:100], y_target: y_training_labels[0:100]}), "\n"

            plt.title("Learning Curve using cross entropy cost function") #Change cross_entropy to mean_squared error
            print "Cost: ", error, "\n"
            plt.xlabel("Number of Epochs")
            plt.ylabel("Cost")
            plt.plot(epochs, errors)
            plt.show()

            #print "TEST", session.run(y_estimated, feed_dict={x: x_training_data[130:132], y_target: y_training_labels[130:132]}), "\n"

            break
correct_prediction = tf.equal(tf.argmax(y_estimated, 1), tf.argmax(y_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print "Test Data Accuracy ", (accuracy.eval(feed_dict={x: x_training_data[100:140], y_target: y_training_labels[100:140]}))
#print y_training_labels[130:132]

plt.title("Learning Curve using cross entropy cost function") #Change cross_entropy to mean_squared error
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(epochs, errors)
plt.show()