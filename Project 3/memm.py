"""
Definition of a Maximum Entropy Markov Model
Diep (Emma) Vu
I affirm that I have carried out my academic endeavors with full
academic honesty.
"""


class MEMM:
    def __init__(self, states, vocabulary, vectorizer, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        # TODO: Complete this method
        self.states = states #state classifier predict
        self.vocabulary = dict((vocabulary[i], i) for i in range(len(vocabulary)))
        self.vectorizer = vectorizer
        self.classifier = classifier


    # TODO: Add additional methods that are needed. In particular, you
    # will need a method that can take a dictionary of features
    # representing a word and the tag chosen for the previous word and
    # return the probabilities of each of the MEMM's states.
    def vectorize_feats(self, features, prev_state):
        """Returns the vectorized version of the given features."""
        features['prev_state'] = prev_state  # Add the previous tag to the features dictionary
        return self.vectorizer.transform(features) 
    
    def predict_log_proba(self, features, prev_state):
        """Returns the probabilities of each of the MEMM's states."""
        vector_feat = self.vectorize_feats(features, prev_state)
        return self.classifier.predict_log_proba(vector_feat)[0] #shape num state x 1




    
    

