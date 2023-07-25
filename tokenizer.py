import argparse
import os
import pickle
import re

import pandas as pd
import pefile
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def extract_strings(file):
    all_strings = []

    try:
        pe = pefile.PE(file)

        sections = pe.sections
        for section in sections:
            data = section.get_data()
            ascii_strings = re.findall(b"[!-~]{5,}", data)
            unicode_strings = re.findall(b"(?:[\x20-\x7E][\x00]){5,}", data)

            for string in ascii_strings:
                all_strings.append(string.decode('ascii'))

            for string in unicode_strings:
                all_strings.append(string.decode('utf-16'))

    except Exception as e:
        print(f"An error occurred: {str(e)}; file: {file}")

    return ' '.join(all_strings)

def get_files_from_directory(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def build_features(malware_directory, benign_directory):
    malware_files = get_files_from_directory(malware_directory)
    benign_files = get_files_from_directory(benign_directory)

    all_files = [(file, 0) for file in malware_files] + [(file, 1) for file in benign_files]

    rows = []
    for file, label in all_files:
        strings = extract_strings(file)
        rows.append({'strings': strings, 'label': label})

    return pd.DataFrame(rows)

def build_and_test_model(df):
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['strings'].values)

    # Save tokenizer to file
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X = tokenizer.texts_to_sequences(df['strings'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df['label']).values

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1)
    
    model.save('malware_detection_model.h5')
    
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Compute precision, recall, F1-score and support
    print(classification_report(Y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=["Malware", "Benign"]))

    score, accuracy = model.evaluate(X_test, Y_test, verbose = 2, batch_size = 64)
    print(f"Model accuracy: {accuracy}")

# Command-line arguments parsing
parser = argparse.ArgumentParser(description='Malware classification script')
parser.add_argument('-m','--malware', help='Directory with malware files', required=True)
parser.add_argument('-b','--benign', help='Directory with benign files', required=True)

args = vars(parser.parse_args())

df = build_features(args['malware'], args['benign'])
build_and_test_model(df)
