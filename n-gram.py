#!/usr/bin/env python3

__author__ = 'Matthew Schwartz (@schwartz1375)'

import argparse
import os
import re

import pandas as pd
import pefile
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
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

    # Use CountVectorizer for feature extraction with n-grams
    # We're setting ngram_range to (1, 2) to include both 1-grams and 2-grams,
    # but you can adjust this depending on your specific needs
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train['strings'])
    X_test = vectorizer.transform(test['strings'])
    y_train = train['label']
    y_test = test['label']

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Print the classification report
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")


if __name__ == '__main__':
    main()
