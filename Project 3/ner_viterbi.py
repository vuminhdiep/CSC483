"""Named Entity Recognition as a sequence tagging task.

Author: Kristina Striegnitz and Diep (Emma) Vu

I affirm that I have carried out my academic endeavors with full
academic honesty.

Complete this file for part 2 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import math
import numpy as np
from memm import MEMM
import time

#################################
#
# Word classifier
#
#################################

def getfeats(word, o):
    """Take a word its offset with respect to the word we are trying to
    classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        # (o + 'lower', word.lower()),
        # (o + 'upper', word.upper()),
        #(o + 'hyphen', contain_hyphen(word)),
        # (o + 'digit', contain_digits(word)),
        #(o + 'shape', word_shape(word)),
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


#################################
#
# Viterbi decoding
#
#################################

def viterbi(obs, memm, pretty_print=False):
    """The Viterbi algorithm. Calculates what sequence of states is most
    likely to produce the given sequence of observations.

    V is the main data structure that represents the trellis/grid of
    Viterbi probabilities. It is a list of np.arrays. Each np.array
    represents one column in the grid.

    The variable 'path' maintains the currently most likely paths. For
    example, once we have finished processing the third observation,
    then 'path' contains for each possible state, the currently most
    likely path that leads to that state given these three
    observations.
    """
    V = []
    paths =  np.array([[i] for i in range(len(memm.states))])
    V.append(memm.predict_log_proba(obs[0], '<S>'))
    
    # Run Viterbi for all of the subsequent steps/observations: t > 0.
    for t in range(1,len(obs)):
        log_probs = V[t - 1] + np.array([memm.predict_log_proba(obs[t], prev_state) for prev_state in memm.states]).T #get the multiplication of the prob from classifier
        
        V.append(np.max(log_probs, axis=1)) #get max of log prob, 
        max_prev_states = np.argmax(log_probs, axis=1)
        paths = np.insert(paths[max_prev_states], t, range(len(memm.states)), axis=1)
        
    if pretty_print:
        pretty_print_trellis(V)

    most_probable_final_state = np.argmax(V[len(obs)-1])
    most_probable_path = [memm.states[i] for i in paths[most_probable_final_state]]
    return most_probable_path
    
 

def pretty_print_trellis(V):
    """Prints out the Viterbi trellis formatted as a grid."""
    print("    ", end=" ")
    for i in range(len(V)):
        print("%7s" % ("%d" % i), end=" ")
    print()
 
    for y in V[0].keys():
        print("%.5s: " % y, end=" ")
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=" ")
        print()

def logprob(p):
    """Returns the logarithm of p."""
    if p != 0:
        return math.log(p)
    else:
        return float('-inf')


if __name__ == "__main__":
    start = time.time()
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []
    vocab = set() 

    for sent in train_sents:
        prev_state = '<S>'
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            # TODO: training needs to take into account the label of
            # the previous word. And <S> if i is the first words in a
            # sentence.
            curr_state = sent[i][-1]
            feats['prev_state'] = prev_state
            train_feats.append(feats)
            train_labels.append(curr_state)
            prev_state = curr_state
            vocab.add(sent[i][0])

    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats) #try to fit on matrix of feature list
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.

    model = LogisticRegression(max_iter=400)
    model.fit(X_train, train_labels)

    memm = MEMM(model.classes_, list(vocab), vectorizer, model)

    print("\nTesting ...")
    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_memm.txt. That is the results_memm.txt you should hand
    # in.
    y_pred = []
    test_labels = []
    
    for sent in dev_sents:
        # TODO: extract the feature representations for the words from
        # the sentence; use the viterbi algorithm to predict labels
        # for this sequence of words; add the result to y_pred
        #each observation = feature list of a word, no need encode word
        obs_list = []
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            obs_list.append(feats)
            test_labels.append(sent[i][-1])
        X_test = viterbi(obs_list, memm)
        y_pred.extend(X_test)

        

    print("Writing to results_memm.txt")
    # format is: word gold pred
    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")
    end = time.time()
    print("Time elapsed:", end - start)
    print("Now run: python3 conlleval.py results_memm.txt")






