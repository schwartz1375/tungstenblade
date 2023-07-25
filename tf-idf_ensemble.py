#!/usr/bin/env python3

__author__ = 'Matthew Schwartz (@schwartz1375)'

import argparse
import os
import re
import uuid

import pandas as pd
import pefile
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


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


def main():
    # generate a unique identifier for the training directory
    train_dir = 'catboost_info_' + str(uuid.uuid4())

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'malware_dir', help='Directory of malware binary files')
    parser.add_argument('benign_dir', help='Directory of benign binary files')
    args = parser.parse_args()

    # Extract strings from the binary files
    malware_files = os.listdir(args.malware_dir)
    malware_strings = [extract_strings(os.path.join(
        args.malware_dir, file)) for file in malware_files]
    benign_files = os.listdir(args.benign_dir)
    benign_strings = [extract_strings(os.path.join(
        args.benign_dir, file)) for file in benign_files]

    # Create a DataFrame
    malware_df = pd.DataFrame(
        {'filename': malware_files, 'strings': malware_strings, 'label': 'malware'})
    benign_df = pd.DataFrame(
        {'filename': benign_files, 'strings': benign_strings, 'label': 'benign'})
    df = pd.concat([malware_df, benign_df])

    # Split into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Use TF-IDF for feature extraction
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train['strings'])
    X_test = vectorizer.transform(test['strings'])
    y_train = train['label']
    y_test = test['label']

    # Train a Random Forest classifier
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)

    # Train a CatBoost classifier
    clf_cb = CatBoostClassifier(
        verbose=50, random_state=42, train_dir=train_dir)
    clf_cb.fit(X_train, y_train)

    # Get the predictions and probabilities for each classifier
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_cb = clf_cb.predict(X_test)
    y_prob_rf = clf_rf.predict_proba(X_test)
    y_prob_cb = clf_cb.predict_proba(X_test)

    # Combine the predictions: if the classifiers disagree, choose the prediction with higher confidence
    y_pred = []
    for i in range(len(y_pred_rf)):
        if y_pred_rf[i] == y_pred_cb[i]:
            y_pred.append(y_pred_rf[i])
        else:
            if max(y_prob_rf[i]) > max(y_prob_cb[i]):
                y_pred.append(y_pred_rf[i])
            else:
                y_pred.append(y_pred_cb[i])

    # Print the classification report
    print(classification_report(y_test, y_pred))

    # Print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")


if __name__ == '__main__':
    main()
