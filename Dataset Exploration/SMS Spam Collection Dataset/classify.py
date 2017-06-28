import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

train_x, train_y = np.loadtxt('train_x.txt'), np.loadtxt('train_y.txt')
features_length = train_x.shape[1]
print(features_length)

test_x, test_y = np.loadtxt('test_x.txt'), np.loadtxt('test_y.txt')

model = Sequential()
model.add(Dense(100, input_shape=(features_length,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=100, epochs=50)
preds = model.predict_classes(test_x)
score = model.evaluate(test_x, test_y)

print(score)


