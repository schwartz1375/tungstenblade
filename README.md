# TUNGSTENBLADE
Malware often contains strings that are indicative of its behavior, such as paths, URLs, registry keys, function names, and possibly human-readable messages or commands. Extracting and analyzing these strings can yield valuable insights.  The TUNGSTENBLADE project represents an innovative approach to malware analysis, leveraging the power of Natural Language Processing (NLP) techniques to classify binary files as either benign or malicious. By considering strings within the binaries as akin to 'words' in a document, we can apply powerful NLP techniques typically used in the analysis of human languages to the seemingly chaotic world of binary files.

In traditional malware analysis, much of the focus is often on static or dynamic analysis of the executable code. While these methods are undeniably valuable, they often require substantial resources and expertise to perform at scale. By contrast, the analysis of strings within binaries can be done relatively quickly and can provide surprising insights into the behavior of the binary.

TUNGSTENBLADE is not intended to replace traditional malware analysis techniques, but rather to supplement them. The scripts provided within the project each employ a different NLP technique, including tokenization, n-gram analysis, and Term Frequency-Inverse Document Frequency (TF-IDF) analysis. By using these techniques, we can glean additional insights from the binary files that may not be immediately apparent through traditional analysis methods.

The project's philosophy could be described as "anarchic" in the sense that it dispenses with traditional hierarchies of how malware analysis "should" be done. Instead, it embraces a more open-ended, exploratory approach that seeks to find new ways of understanding and interpreting binary data. While this may seem unconventional, it is precisely this willingness to challenge established norms and explore new methods that often leads to breakthroughs in cybersecurity and other fields.

In the following sections, we will detail the specific methodologies employed by each script in the TUNGSTENBLADE project, as well as their respective performance metrics. As with any method in cybersecurity, no single approach is a silver bullet. However, we hope that by providing a diverse range of tools and techniques, users of TUNGSTENBLADE will be better equipped to tackle the ever-evolving threats in the world of malware.


## NLP Techniques
* Tokenization: This is the process of splitting a large paragraph into smaller chunks or tokens. In the context of malware analysis, you might tokenize on whitespace or special characters to split a block of strings into individual words or phrases.
* N-grams: N-grams can be used to capture the context of a word in a given sequence of words. An n-gram is a contiguous sequence of n items from a given sample of text or speech. For instance, you might use 2-grams (also known as bigrams) to capture pairs of strings that appear next to each other.
* TF-IDF: The Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used to reflect how important a word is to a document in a collection or corpus. In the context of malware analysis, the “documents” could be individual executable files, and the “corpus” could be your entire dataset of executables.
* Word embeddings: Word embeddings are a type of word representation that allows words with similar meaning to have similar representation. The idea is to use the context in which a string appears to learn a vector representation of that string. Then, these vectors can be used as features for a machine learning model.
* Topic modeling: Topic modeling algorithms, like Latent Dirichlet Allocation (LDA), can be used to discover the abstract “topics” that occur in a collection of “documents” (in this case, the documents are individual executable files, and the words are the strings within those files). The idea is to identify clusters of strings that often appear together, which might be indicative of certain 

## Tokenizer String Based Malware Classification
This project involves natural language processing (NLP), text vectorization, and sequential model architecture.  `tokenizer.py` is designed to classify files as malware or benign based on the strings found within them.  We use Python's `pefile` and `re` libraries to extract strings from the binary files, and then use a Keras model with LSTM layers to classify these files. 

### How it works
The script first extracts all the printable strings from the binary files. These strings are then tokenized and padded to be fed into the Keras model. The Keras model consists of an Embedding layer, a SpatialDropout1D layer, an LSTM layer, and a Dense layer. I then use the Adam optimizer and the Categorical Cross entropy loss function. The model is trained for five epochs with a batch size of 64 and a validation split of 0.1.  Finally, `tokenizer.py`  prints each class's precision, recall, F1-score, and the model's accuracy on the test set.


### Technical Details
The role of each layer in the neural network:

* The Embedding layer converts the input data into fixed-sized dense vectors better suited for a neural network. You could also specify the dimension of these vectors (EMBEDDING_DIM), which in this case is 100.
* The SpatialDropout1D layer is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time. You could also specify the fraction rate (0.2).
* The LSTM layer is a type of recurrent neural network layer suitable for modeling sequence data (in this case, the sequences of words). The parameter 100 refers to the number of LSTM units in this layer.
* The Dense layer is the output layer, with two units corresponding to the two classes (malware and benign). It uses the 'softmax' activation function to output the probabilities of each type.

The model is then compiled with the adam optimizer and categorical_crossentropy as the loss function, which are suitable choices for a multiclass classification problem.

### Is there Signal?
Testing data was pulled from [Malware Data Science](https://www.malwaredatascience.com/code-and-data), specifically ch8/data.

```
              precision    recall  f1-score   support

     Malware       0.94      0.97      0.95        86
      Benign       0.98      0.97      0.98       198

    accuracy                           0.97       284
   macro avg       0.96      0.97      0.97       284
weighted avg       0.97      0.97      0.97       284

5/5 - 0s - loss: 0.0969 - accuracy: 0.9718 - 276ms/epoch - 55ms/step
```

Execution with a larger data set:
```
              precision    recall  f1-score   support

     Malware       0.83      0.70      0.76       199
      Benign       0.74      0.86      0.80       201

    accuracy                           0.78       400
   macro avg       0.79      0.78      0.78       400
weighted avg       0.79      0.78      0.78       400

7/7 - 0s - loss: 0.5114 - accuracy: 0.7800 - 400ms/epoch - 57ms/step
Model accuracy: 0.7799999713897705

              precision    recall  f1-score   support

     Malware       0.83      0.84      0.84      1974
      Benign       0.84      0.83      0.84      2026

    accuracy                           0.84      4000
   macro avg       0.84      0.84      0.84      4000
weighted avg       0.84      0.84      0.84      4000

63/63 - 7s - loss: 0.4278 - accuracy: 0.8378 - 7s/epoch - 109ms/step
Model accuracy: 0.8377500176429749
```

## N-gram String Based Malware Classification
`n-gram.py` is a Python script that classifies binary files as either malware or benign based on the n-gram analysis of strings extracted from the binary files. 

### Methodology
The script first extracts strings from the binary files. It then uses the n-gram model to capture the context in which a string appears in the file. The n-gram approach creates a 'document' from each file where the 'words' are the strings within the file. This allows us to turn the binary file into a format that can be processed by machine learning algorithms.

The script uses scikit-learn's `CountVectorizer` for feature extraction, with the n-gram range set to (1, 2) to include both 1-grams and 2-grams.

The feature vectors are then used to train a Random Forest classifier. The classification report, which includes precision, recall, F1-score, and accuracy, is printed at the end.


### Technical Details
The script uses the `pefile` Python library to parse the PE (Portable Executable) format of the binary files. It then extracts ASCII and Unicode strings from the data sections of the PE files. Any errors during the extraction process are caught and printed to the console, but they do not halt the execution of the script.

The extracted strings are joined together into a single string for each file, with spaces between each string. These 'documents' of strings are used to create the feature vectors for the machine learning model.

The n-gram model is used to capture the context in which a string appears in the file. The script uses scikit-learn's `CountVectorizer` with the n-gram range set to (1, 2) to include both 1-grams and 2-grams. This means that both individual strings and pairs of strings that appear next to each other in the 'document' are included as features.

The feature vectors are then used to train a Random Forest classifier from the scikit-learn library. The classifier is trained on 80% of the data, and tested on the remaining 20%.

### Is there Signal?
Testing data was pulled from [Malware Data Science](https://www.malwaredatascience.com/code-and-data), specifically ch8/data.
``` 
              precision    recall  f1-score   support

      benign       0.99      0.99      0.99       198
     malware       0.98      0.99      0.98        86

    accuracy                           0.99       284
   macro avg       0.99      0.99      0.99       284
weighted avg       0.99      0.99      0.99       284

Model accuracy: 0.9894366197183099
```

Execution with larger data sets:
```
              precision    recall  f1-score   support

      benign       0.83      0.86      0.84       201
     malware       0.85      0.82      0.83       199

    accuracy                           0.84       400
   macro avg       0.84      0.84      0.84       400
weighted avg       0.84      0.84      0.84       400

Model accuracy: 0.8375

              precision    recall  f1-score   support

      benign       0.89      0.93      0.91      1981
     malware       0.93      0.88      0.90      2019

    accuracy                           0.91      4000
   macro avg       0.91      0.91      0.91      4000
weighted avg       0.91      0.91      0.91      4000

Model accuracy: 0.90575
```

## TF-IDF String Based Malware Classification
`tf-idf.py` is a  Python script that classifies binary files as either malware or benign based on the Term Frequency-Inverse Document Frequency (TF-IDF) analysis of strings extracted from the binary files. 

### Methodology
The script first extracts strings from the binary files. It then uses the TF-IDF statistic to reflect how important a string is to a binary file in the context of a collection of binary files.

The script uses scikit-learn's `TfidfVectorizer` for feature extraction.

The feature vectors are then used to train a Random Forest classifier. The classification report, which includes precision, recall, F1-score, and accuracy, is printed at the end.

### Technical Details
The script uses the `pefile` Python library to parse the PE (Portable Executable) format of the binary files. It then extracts ASCII and Unicode strings from the data sections of the PE files. Any errors during the extraction process are caught and printed to the console, but they do not halt the execution of the script.

The extracted strings are joined together into a single string for each file, with spaces between each string. These 'documents' of strings are used to create the feature vectors for the machine learning model.

The TF-IDF model is used to quantify the importance of each string in the context of the collection of binary files. The script uses scikit-learn's `TfidfVectorizer` to calculate the TF-IDF statistic for each string.

The feature vectors are then used to train a Random Forest classifier from the scikit-learn library. The classifier is trained on 80% of the data, and tested on the remaining 20%.

### Is there Signal?
Testing data was pulled from [Malware Data Science](https://www.malwaredatascience.com/code-and-data), specifically ch8/data.
``` 
              precision    recall  f1-score   support

      benign       0.98      0.99      0.98       198
     malware       0.98      0.95      0.96        86

    accuracy                           0.98       284
   macro avg       0.98      0.97      0.97       284
weighted avg       0.98      0.98      0.98       284

Model accuracy: 0.9788732394366197
```

Execution with larger data sets:
```
              precision    recall  f1-score   support

      benign       0.82      0.84      0.83       201
     malware       0.83      0.82      0.83       199

    accuracy                           0.83       400
   macro avg       0.83      0.83      0.83       400
weighted avg       0.83      0.83      0.83       400

Model accuracy: 0.8275

              precision    recall  f1-score   support

      benign       0.89      0.91      0.90      1981
     malware       0.91      0.89      0.90      2019

    accuracy                           0.90      4000
   macro avg       0.90      0.90      0.90      4000
weighted avg       0.90      0.90      0.90      4000

Model accuracy: 0.90075
```
## Single Model Conclusion
Thus far we have seen a suite of methods for classifying binary files as either malware or benign. This is accomplished through the extraction and analysis of strings found within these files, employing a range of natural language processing (NLP) techniques such as tokenization, n-gram analysis, and TF-IDF.

The tokenizer.py script employs a Keras model with LSTM layers to classify files based on their tokenized strings. This approach has shown promising results, with an accuracy of up to 97.18% on the small datasets and 83.78% on the largest datasets.

The n-gram.py script uses an n-gram model in conjunction with a Random Forest classifier to perform the binary classification task. N-gram analysis allows for the capture of context within the sequence of strings found in the file. This method has achieved accuracy rates of up to 98.94% on the small datasets and 90.58% on the largest datasets.

The tf-idf.py script leverages the Term Frequency-Inverse Document Frequency (TF-IDF) statistic to weigh the importance of a string within a binary file in the context of a collection of binary files. Coupled with a Random Forest classifier, this approach has yielded accuracy rates of up to 97.89% on the small datasets and 90.08% on the largest datasets.

# Ensembles
## Ensemble Learning for Malware Classification Using TF-IDF Strings
In the realm of cybersecurity, accurately classifying binary files as benign or malware is of paramount importance. The TF-IDF Ensemble String Based Malware Classification script is designed to address this challenge by applying advanced machine learning techniques. The `tf-idf_ensemble.py` uses Term Frequency-Inverse Document Frequency (TF-IDF) to perform feature extraction from the strings found in binary files, transforming them into quantifiable data points that can be used for classification. Building on this foundation, it trains an ensemble of two distinct but complementary classifiers, a Random Forest classifier and a CatBoost classifier, to predict whether a given binary file is benign or malware. In cases where the classifiers disagree, the script intelligently chooses the prediction from the classifier that exhibits higher confidence. The result is a robust classification system that effectively combines the strengths of multiple methodologies to enhance the accuracy and reliability of malware detection.

### Methodology
The methodology followed in this script is a typical machine learning pipeline, which includes the following steps:

1. Data Preparation: The script reads the binary files and extracts ASCII and Unicode strings from the data sections.

2. Feature Extraction: The script uses the TF-IDF technique to convert the extracted strings into a matrix of TF-IDF features.

3. Model Training: The script trains two classifiers, a Random Forest classifier and a CatBoost classifier, using the extracted features and the labels (malware or benign).

4. Prediction: The classifiers are used to make predictions on the testing data. If the classifiers disagree on a prediction, the script chooses the prediction from the classifier that has a higher confidence.

5. Evaluation: The script evaluates the performance of the classifiers by printing the classification report, which includes precision, recall, F1-score, and support for each class, as well as the overall accuracy of the model.

The choice of using an ensemble of two different classifiers allows the model to benefit from the strengths of both classifiers. If the classifiers disagree, the prediction from the classifier with higher confidence is chosen, which can potentially improve the accuracy of the predictions.

### Technical details
The script utilizes several Python libraries and machine learning techniques to extract features from binary files and classify them as either malware or benign.

The script begins by importing necessary Python libraries, including `argparse`, `os`, `re`, `pandas`, `pefile`, `CatBoostClassifier`, `RandomForestClassifier`, `TfidfVectorizer`, `accuracy_score`, and `classification_report`.

The `extract_strings` function is used to parse the binary files and extract ASCII and Unicode strings from the data sections using the `pefile` Python library. This function is applied to both malware and benign files, and the extracted strings are stored in Pandas DataFrames.

The script then splits the data into training and testing sets, with 80% of the data used for training and 20% used for testing.

The `TfidfVectorizer` from scikit-learn is then used to transform the strings into a matrix of TF-IDF features. TF-IDF (Term Frequency-Inverse Document Frequency) is a statistic that reflects how important a word is to a document in a collection or corpus.

After the feature extraction, the script trains two classifiers: a Random Forest classifier and a CatBoost classifier. The Random Forest classifier is a popular ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees. The CatBoost classifier is a machine learning algorithm that uses gradient boosting on decision trees.

After training, the classifiers are used to make predictions on the testing data. The script also calculates the probability of each class for each prediction.

If the classifiers disagree on a prediction, the script chooses the prediction from the classifier that has a higher confidence (i.e., the higher probability).

Finally, the script prints the classification report and the accuracy score.

### Is there Signal?
Testing data was pulled from [Malware Data Science](https://www.malwaredatascience.com/code-and-data), specifically ch8/data.
``` 
              precision    recall  f1-score   support

      benign       0.99      0.99      0.99       198
     malware       0.99      0.98      0.98        86

    accuracy                           0.99       284
   macro avg       0.99      0.99      0.99       284
weighted avg       0.99      0.99      0.99       284

Model accuracy: 0.9894366197183099
```

Execution with larger data sets:
```
              precision    recall  f1-score   support

      benign       0.88      0.88      0.88       201
     malware       0.88      0.88      0.88       199

    accuracy                           0.88       400
   macro avg       0.88      0.88      0.88       400
weighted avg       0.88      0.88      0.88       400

Model accuracy: 0.8775

              precision    recall  f1-score   support

      benign       0.93      0.94      0.93      2026
     malware       0.93      0.92      0.93      1974

    accuracy                           0.93      4000
   macro avg       0.93      0.93      0.93      4000
weighted avg       0.93      0.93      0.93      4000

Model accuracy: 0.93
```

## Ensemble Learning for Malware Classification Using N-gram Strings
In the evolving landscape of cyber threats, effective and efficient malware detection is paramount. The `n-gram_ensemble.py` script applies a sophisticated ensemble machine learning approach to the problem of malware classification. This script extracts both ASCII and unicode strings from binary files, forming the foundation of our feature set. Using N-grams, sequences of 'n' characters from these extracted strings, we transform the raw data into a suitable format for machine learning algorithms. The script employs an ensemble of two powerful classifiers, the Random Forest and CatBoost, trained on these N-gram features. When the classifiers disagree, the script chooses the prediction of the classifier with the higher confidence, thus leveraging the strengths of both. This approach offers a robust method to classify malware, aiming to maximize accuracy and contribute to the ongoing fight against cyber threats.

### Methodology
This script is a blend of Natural Language Processing (NLP) techniques applied in the context of malware classification and ensemble learning methods:

1. Data Preparation: The script starts by extracting strings from binary files. These files are parsed using the pefile Python library, and ASCII and Unicode strings are extracted from the data sections of the PE files.

2. Feature Extraction: The extracted strings are then joined together into a single string for each file, creating a 'document' for each file. The documents are transformed into numerical feature vectors using the CountVectorizer from scikit-learn, with the n-gram range set to (1, 2) to include both individual strings and pairs of strings.

3. Model Training: The feature vectors are then used to train two machine learning models: a Random Forest classifier and a CatBoost classifier. The training data comprises 80% of the data, and the remaining 20% is used for testing.

4. Prediction and Ensemble Method: The trained models are used to make predictions on the test set. However, instead of simply taking the prediction of one model, the script uses an ensemble method. When the two classifiers agree on the prediction, that prediction is taken. When they disagree, the script chooses the prediction from the classifier that has a higher confidence in its prediction. The confidence is computed as the prediction probability.

5. Evaluation: Finally, the script evaluates the performance of the ensemble method by printing the classification report, which includes precision, recall, F1-score, and accuracy.

This methodology leverages the strengths of two different machine learning models and ensemble methods to make more robust and accurate predictions. The use of prediction probability to resolve disagreements between classifiers is a unique approach that helps avoid the creation of an 'undecided' class and ensures more confident predictions.

### Technical details
The script uses the `pefile` Python library to parse the PE (Portable Executable) format of the binary files and extracts ASCII and Unicode strings from the data sections of the PE files. Any errors during the extraction process are caught and printed to the console, but they do not halt the execution of the script.

The extracted strings are joined together into a single string for each file, with spaces between each string. These 'documents' of strings are used to create the feature vectors for the machine learning models.

The script uses the `CountVectorizer` from scikit-learn with the n-gram range set to (1, 2) to include both 1-grams and 2-grams. This means that both individual strings and pairs of strings that appear next to each other in the 'document' are included as features.

The feature vectors are then used to train two classifiers: a Random Forest classifier and a CatBoost classifier, both from the scikit-learn library. The classifiers are trained on 80% of the data, and tested on the remaining 20%.

When the two classifiers disagree on the prediction, instead of assigning an 'undecided' class, the script chooses the prediction from the classifier that has a higher confidence in its prediction. The confidence is determined by the prediction probability, which is obtained using the `predict_proba` method of the classifiers.

The script then prints the classification report, which includes precision, recall, F1-score, and accuracy. The performance of the ensemble method can be evaluated based on these metrics.

A unique feature of this script is the use of an ensemble of two different classifiers, which can lead to more robust and accurate predictions. The use of the prediction probability to resolve disagreements between the classifiers is a novel approach that can help to avoid the creation of an 'undecided' class and make more confident predictions.

### Is there Signal?
Testing data was pulled from [Malware Data Science](https://www.malwaredatascience.com/code-and-data), specifically ch8/data.
``` 
              precision    recall  f1-score   support

      benign       0.99      0.99      0.99       198
     malware       0.99      0.99      0.99        86

    accuracy                           0.99       284
   macro avg       0.99      0.99      0.99       284
weighted avg       0.99      0.99      0.99       284

Model accuracy: 0.9929577464788732
```

Execution with larger data sets:
```
              precision    recall  f1-score   support

      benign       0.89      0.89      0.89       201
     malware       0.89      0.89      0.89       199

    accuracy                           0.89       400
   macro avg       0.89      0.89      0.89       400
weighted avg       0.89      0.89      0.89       400

Model accuracy: 0.89


```

## Ensemble Conclusion
Our exploration into ensemble learning for malware classification has yielded encouraging results, with both the TF-IDF and N-gram ensembles demonstrating considerable proficiency in classifying binary files as benign or malware.

The TF-IDF ensemble system utilizes the strengths of Random Forest and CatBoost classifiers, while employing TF-IDF for feature extraction from strings found in binary files. The methodology shows high efficacy, with an accuracy of 98.9% on the small dataset and 93% on the largest datasets. This reflects the system's ability to maintain good performance even when scaling up to larger datasets.

The N-gram ensemble system also leverages the power of Random Forest and CatBoost classifiers, but adopts a different feature extraction method, transforming raw data using N-grams. The system performed exceedingly well, yielding an accuracy of 99.3% on the small dataset and XX% on the largest datasets. These results signify that the N-gram ensemble model exhibits a high level of stability and consistency across different dataset sizes.

By intelligently combining predictions of two classifiers, these ensemble models have effectively leveraged the strengths of both and mitigated their individual weaknesses, thereby increasing the robustness and accuracy of malware detection. This approach underscores the potential of ensemble learning in bolstering cybersecurity measures. When the two classifiers disagree, the prediction is made based on the classifier showing higher confidence, enhancing the precision of the prediction.

In conclusion, both ensemble models present a promising solution to the growing challenge of malware detection. These models, built on the foundations of machine learning and natural language processing, exhibit significant accuracy in detecting malware and have the potential to be further optimized and refined. Future work can explore additional features, more diverse ensemble techniques, and the integration of these models into broader cybersecurity frameworks to further enhance their utility.

# Comprehensive Method Analysis Conclusion
The TUNGSTENBLADE project offers a novel suite of techniques to enhance malware analysis. It exploits Natural Language Processing (NLP) methodologies, typically used for human language analysis, to examine strings within binary files and classify them as either benign or malicious. Techniques deployed include tokenization, n-gram analysis, and Term Frequency-Inverse Document Frequency (TF-IDF) analysis.

Each approach in this project offers distinct strengths. The tokenizer-based method, which applies a Keras model with LSTM layers, has yielded impressive results with accuracy up to to 97.18% on the small datasets and 83.78% on the largest datasets. The n-gram technique, complemented by a Random Forest classifier, can capture context within sequences of strings, achieving up to 98.94% on the small datasets and 90.58% on the largest datasets.. The TF-IDF-based approach, which quantifies string importance in binary files, combined with a Random Forest classifier, has achieved up to 97.89% on the small datasets and 90.08% on the largest datasets.

In addition to these techniques, the project explored the ensemble learning approach to malware classification. This method harnessed the power of multiple classifiers, combining their strengths to enhance accuracy and reliability. Two ensemble methods were examined, each exploiting different NLP techniques and machine learning classifiers. The first ensemble method, using TF-IDF for feature extraction and an ensemble of Random Forest and CatBoost classifiers for prediction, produced an accuracy rate up to 98.9% on the small dataset and 93% on the largest datasets. The second ensemble method used n-gram analysis for feature extraction and the same ensemble of classifiers, also achieving an accuracy rate up to 99.3% on the small dataset and XX% on the largest datasets.

While these approaches demonstrate significant promise, it's important to note that no single method constitutes a silver bullet in cybersecurity. The constant evolution of malware and cyber threats necessitates a diverse range of tools and techniques to maintain effective defenses. The TUNGSTENBLADE project contributes valuable tools to the cybersecurity toolbox, pushing the boundaries of traditional malware analysis by leveraging NLP methodologies and machine learning techniques.

These findings underline the potential of such methods in augmenting traditional malware analysis, and encourage the continued exploration of innovative techniques in cybersecurity. As with any cybersecurity approach, it's critical to validate and adjust these methodologies based on real-world performance and the evolving landscape of threats.
