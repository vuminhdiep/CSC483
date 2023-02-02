"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Diep (Emma) Vu

As a student at Union College, I am part of a community that values intellectual effort, curiosity and discovery. I understand that in order to truly claim my educational and academic achievements, I am obligated to act with academic integrity. Therefore, I affirm that I will carry out my academic endeavors with full academic honesty, and I rely on my fellow students to do the same.

Complete this file for parts 2-4 of the project.

"""

import time
from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate, get_confusion_matrix

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, labels = load_file(data_file)
    y_pred = [1] * len(words)
    evaluate(y_pred, labels)


### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    train_x, y_train = load_file(training_file)
    dev_x, y_dev = load_file(development_file)
    best_threshold = 0
    best_fscore = 0
    for threshold in range(1, 10):
        train_pred = [1 if len(word) >= threshold else 0 for word in train_x]
        dev_pred = [1 if len(word) >= threshold else 0 for word in dev_x]

        curr_fscore = get_fscore(train_pred, y_train)
        if curr_fscore > best_fscore:
            best_threshold = threshold
            best_fscore = curr_fscore
    
    train_pred = [1 if len(word) >= best_threshold else 0 for word in train_x]
    dev_pred = [1 if len(word) >= best_threshold else 0 for word in dev_x]
    print("Best threshold:", best_threshold)
    print("Performance on training data")
    evaluate(train_pred, y_train)
    print("\nPerformance on development data")
    evaluate(dev_pred, y_dev)


### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            #token = token.lower()   #lower all the tokens to avoid case sensitive duplicate counts, for better counting frequency 
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    start_time = time.time()
    train_x, y_train = load_file(training_file)
    dev_x, y_dev = load_file(development_file)

    min_freq = min(counts.values())
    max_freq = max(counts.values())

    start = 1000000
    end = 100000000

    best_threshold = 0
    best_fscore = 0
    
    for threshold in range(start, end, 100000):
        train_pred = [1 if counts.get(word, 0) <= threshold else 0 for word in train_x]
        dev_pred = [1 if counts.get(word, 0) <= threshold else 0 for word in dev_x]

        curr_fscore = get_fscore(train_pred, y_train)
        if curr_fscore > best_fscore:
            best_threshold = threshold
            best_fscore = curr_fscore
    
    train_pred = [1 if counts.get(word, 0) <= best_threshold else 0 for word in train_x]
    dev_pred = [1 if counts.get(word, 0) <= best_threshold else 0 for word in dev_x]
    print("Best threshold:", best_threshold)
    print("Performance on training data")
    evaluate(train_pred, y_train)
    print("\nPerformance on development data")
    evaluate(dev_pred, y_dev)
    end_time = time.time()

    print("Time taken:", end_time - start_time)


### 3.1: Naive Bayes

def length_feature(words):
    """Return a list of length features for each word in the list"""
    lengths = []
    for i in range(len(words)):
        lengths.append(len(words[i]))
    return lengths

def frequency_feature(words, counts):
    """Return a list of frequency features for each word in the list"""
    frequencies = []
    for word in words:
        frequencies.append(counts[word])
    return frequencies

def train_feature(words, counts):
    """Extract length and frequency features for each word in the word list for training"""
    length_feats = np.array(length_feature(words))
    length_mean = np.mean(length_feats, axis=0)
    length_std = np.std(length_feats, axis=0)

    freq_feats = np.array(frequency_feature(words, counts))
    freq_mean = np.mean(freq_feats, axis=0)
    freq_std = np.std(freq_feats, axis=0)

    #Normalize features
    length_feats = (length_feats - length_mean) / length_std
    freq_feats = (freq_feats - freq_mean) / freq_std

    features_train_stats = np.c_[length_feats, freq_feats], [length_mean, length_std, freq_mean, freq_std] #concatenate by column to become a 2D array
    return features_train_stats
    

def test_feature(words, counts, train_stats):
    """Extract length and frequency features for each word in the word list for testing"""
    length_feats = np.array(length_feature(words))
    freq_feats = np.array(frequency_feature(words, counts))
    
    length_mean = train_stats[0] 
    length_std = train_stats[1]
    freq_mean = train_stats[2]
    freq_std = train_stats[3]

    #Normalize features
    length_feats = (length_feats - length_mean) / length_std 
    freq_feats = (freq_feats - freq_mean) / freq_std 

    features_test = np.c_[length_feats, freq_feats] #concatenate by column to become 2D array
    return features_test

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    
    train_x, train_stats = train_feature(words_train, counts)
    clf = GaussianNB()
    clf.fit(train_x, labels_train)

    dev_x = test_feature(words_dev, counts, train_stats)
    y_pred_train = clf.predict(train_x)
    y_pred_dev = clf.predict(dev_x)

    print("Performance on training data")
    evaluate(y_pred_train, labels_train)
    print("\nPerformance on development data")
    evaluate(y_pred_dev, labels_dev)


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    
    train_x, train_stats = train_feature(words_train, counts)
    clf = LogisticRegression()
    clf.fit(train_x, labels_train)

    dev_x = test_feature(words_dev, counts, train_stats)
    y_pred_train = clf.predict(train_x)
    y_pred_dev = clf.predict(dev_x)

    print("Performance on training data")
    evaluate(y_pred_train, labels_train)
    print("\nPerformance on development data")
    evaluate(y_pred_dev, labels_dev)


### 3.3: Build your own classifier

def syllables_feature(words):
    """Return a list of syllables features for each word in the list"""
    syllables = []
    for i in range(len(words)):
        syllables.append(count_syllables(words[i]))
    return syllables

def synsets_feature(words):
    """Return a list of synsets features for each word in the list"""
    synsets = []
    for word in words:
        synsets.append(len(wn.synsets(word)))
    return synsets


def my_train_feature(words, counts):
    """Extract syllables and synsets features for each word in the word list for training"""
    # Call train_feature function to get length_feats and freq_feats
    features_train_stats = train_feature(words, counts)
    
    syllables_feats = np.array(syllables_feature(words))
    syllables_mean = np.mean(syllables_feats, axis=0)
    syllables_std = np.std(syllables_feats, axis=0)

    synsets_feats = np.array(synsets_feature(words))
    synsets_mean = np.mean(synsets_feats, axis=0)
    synsets_std = np.std(synsets_feats, axis=0)

    # Normalize syllables_feats and synsets_feats
    syllables_feats = (syllables_feats - syllables_mean) / syllables_std
    synsets_feats = (synsets_feats - synsets_mean) / synsets_std

    # Concatenate length_feats, freq_feats, syllables_feats, and synsets_feats
    my_features_train_stats = np.c_[features_train_stats[0], syllables_feats, synsets_feats], np.concatenate((features_train_stats[1], [syllables_mean, syllables_std, synsets_mean, synsets_std]))
    return my_features_train_stats


def my_test_feature(words, counts, train_stats):
    """Extract syllables and synsets features for each word in the word list for testing"""
    # Call test_feature function to get length_feats and freq_feats
    features_test_stats = test_feature(words, counts, train_stats)

    syllables_feats = np.array(syllables_feature(words))
    synsets_feats = np.array(synsets_feature(words))

    syllables_mean = train_stats[4]
    syllables_std = train_stats[5]

    synsets_mean = train_stats[6]
    synsets_std = train_stats[7]

    # Normalize syllables and synsets features
    syllables_feats = (syllables_feats - syllables_mean) / syllables_std
    synsets_feats = (synsets_feats - synsets_mean) / synsets_std

    my_features_test = np.c_[features_test_stats, syllables_feats, synsets_feats]

    return my_features_test


def my_classifier(training_file, development_file, counts):
    """Train a classifier using length, frequency, syllables, and synsets features."""
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)

    models = []
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('Logistic Regression', LogisticRegression()))
    models.append(('Random Forest', RandomForestClassifier()))
    models.append(('Support Vector Machine', SVC()))
    models.append(('Decision Tree', DecisionTreeClassifier()))

    for name, model in models:
        print(name, "\n")
        train_x, train_stats = my_train_feature(words_train, counts)
        clf = model
        clf.fit(train_x, labels_train)

        dev_x = my_test_feature(words_dev, counts, train_stats)
        y_pred_train = clf.predict(train_x)
        y_pred_dev = clf.predict(dev_x)

        print("Performance on training data")
        evaluate(y_pred_train, labels_train)
        
        print("\nPerformance on development data")
        evaluate(y_pred_dev, labels_dev)
        if name == "Support Vector Machine": #error analysis on the best model with development data
            print("\nError analysis on development data")
            error_analysis(y_pred_dev, labels_dev, words_dev)
        print('------------------')
        


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)

def error_analysis(y_pred, y_true, words_dev):
    """Print out the error analysis of the classifier with only top 20 words"""
    count_tp, count_fp, count_tn, count_fn = get_confusion_matrix(y_pred, y_true)
    words_tp, words_fp, words_tn, words_fn = [], [], [], []
    sample_size = 20
    for i, word in enumerate(words_dev):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                if len(words_tp) < sample_size:
                    words_tp.append(word)
            else:
                if len(words_fn) < sample_size:
                    words_fn.append(word)
        else:
            if y_pred[i] == 1:
                if len(words_fp) < sample_size:
                    words_fp.append(word)
            else:
                if len(words_tn) < sample_size:
                    words_tn.append(word)
    print("True Positive: ", count_tp, words_tp)
    print("True Negative: ", count_tn, words_tn)
    print("False Positive: ", count_fp, words_fp)
    print("False Negative: ", count_fn, words_fn)


def my_best_classifier(training_file, development_file, test_file, counts):
    """Train a SVM classifier using combined training and development file and testing on test file."""
    # Load training and development data
    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)

    # Combine training and development data
    words = np.concatenate((words_train, words_dev))
    labels = np.concatenate((labels_train, labels_dev))

    # Train best model on combined data
    train_x, train_stats = my_train_feature(words, counts)
    clf = SVC()
    clf.fit(train_x, labels)

    # Load test data
    words_test, labels_test = load_file(test_file)

    # Predict labels for test data
    test_x = my_test_feature(words_test, counts, train_stats)
    y_pred_test = clf.predict(test_x)

    # Save predicted labels to file
    np.savetxt("test_labels.txt", y_pred_test, fmt="%d")



if __name__ == "__main__":
   
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    print("Loading ngram counts ...")
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)
    
    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE
    my_best_classifier(training_file, development_file, test_file, counts)

