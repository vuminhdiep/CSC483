"""Named Entity Recognition as a classification task.

Author: Kristina Striegnitz and Diep (Emma) Vu

I affirm that I have carried out my academic endeavors with full
academic honesty.
Complete this file for part 1 of the project.
"""
import time
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def getfeats(word, o):
    """Take a word and its offset with respect to the word we are trying
    to classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        # (o + 'lower', word.lower()),
        # (o + 'upper', word.upper()),
        #(o + 'hyphen', contain_hyphen(word)), #not include hyphen increased F1 score
        #(o + 'digit', contain_digits(word)),
        (o + 'shape', word_shape(word)),
        (o + 'short_shape', short_word_shape(word)),
        (o + 'capitalize', word[0].isupper()),

    ]

    #get the prefix and suffix of different character length
    char_length = 4
    for i in range(1, char_length + 1): #maximum length of 4
        if len(word) >= i:
            prefix = word[:i]
            suffix = word[-i:]
            features.append((o + f'prefix{i}', prefix)) 
            features.append((o + f'suffix{i}', suffix))
    return features

def contain_hyphen(word):
    """Check if the word contain any hyphens"""
    return '-' in word

def contain_digits(word):
    """Check if the word contain any digits"""
    for char in word:
        if char.isdigit():
            return True
    return False

def word_shape(word):
    """Represent the abstract letter pattern of the word by mapping lower-case letters to 'x', upper-case to 'X', numbers to 'd', and retaining punctuation"""
    new_word = ''
    for char in word:
        if char.isupper():
            new_word += 'X'
        elif char.islower():
            new_word += 'x'
        elif char.isdigit():
            new_word += 'd'
        else:
            new_word += char
    return new_word

def short_word_shape(word):
    """Represent the abstract letter pattern of the word using shorter word shape features by removing duplicates"""
    shape = word_shape(word)
    # Remove consecutive character types
    new_word = ''
    prev_type = None
    for char in shape:
        if char != prev_type:
            new_word += char
        prev_type = char
    return new_word

    
def word2features(sent, i):
    """Generate all features for the word at position i in the
    sentence. The features are based on the word itself as well as
    neighboring words.
    """
    features = []
    window_size = 3
    # the window around the token (o stands for offset)
    for o in range(-window_size, window_size + 1):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos = sent[i+o][1]
            featlist = getfeats(word, o)
            featlist.append((f'{o}tag', pos)) #add pos tag in the feature list
            features.extend(featlist)
    return features

if __name__ == "__main__":
    start = time.time()
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.
    
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, train_labels)

    print("\nTesting ...")
    test_feats = []
    test_labels = []

    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_classifier.txt. That is the results_classifier.txt you
    # should hand in.
    for sent in test_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    # If you are normaling and/or scaling your training data, make
    # sure to transform your test data in the same way.
    y_pred = model.predict(X_test)

    print("Writing to results_classifier.txt")
    # format is: word gold pred
    j = 0
    with open("results_classifier.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    end = time.time()
    print("Time elapsed:", end - start)
    print("Now run: python3 conlleval.py results_classifier.txt")






