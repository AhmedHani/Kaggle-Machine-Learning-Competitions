import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing import text, sequence

max_features = 2000
maxlen = 450

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("NULL").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("NULL").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_CNN_model():
    embed_size = 128
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(250, activation='sigmoid')(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

model = get_CNN_model()
batch_size = 64
epochs = 2

model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
y_test = model.predict(X_te)

sample_submission = pd.read_csv("sample_submission.csv")

sample_submission[list_classes] = y_test

sample_submission.to_csv("results.csv", index=False)