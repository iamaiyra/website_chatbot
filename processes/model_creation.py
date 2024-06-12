import os
import numpy as np
from keras.api import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_model_from_clean_data():
    # Load cleaned data
    with open(os.path.join('static', 'data', 'cleaned_data.txt'), 'r', encoding='utf-8') as file:
        cleaned_text = file.read()
    print("cleaned data read for model creation")

    # Tokenize the data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned_text])
    total_words = len(tokenizer.word_index) + 1

    # Create sequences
    input_sequences = []
    for line in cleaned_text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Create predictors and label
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = keras.utils.to_categorical(y, num_classes=total_words)

    # Build the model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X, y, epochs=50, verbose=1)

    print("creating model and saving")
    # Save the model
    model.save(os.path.join('static', 'model', 'qa_model.h5'))
    print("model saved")



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    y_test = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1
