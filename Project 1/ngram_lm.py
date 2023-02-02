import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the char '''
    ngrams_list = []
    text = start_pad(c) + text
    for i in range(len(text) - c):
        context = text[i : i + c]
        char = text[i + c]
        ngrams_list.append((context, char))
    return ngrams_list

def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        ''' Initializes the n-gram model with the specified context length c
            and add-k smoothing parameter k '''
        self.c = c
        self.k = k
        self.ngrams = {}
        self.vocab = []
        self.V = 0

    def get_vocab(self):
        ''' Returns the set of chars in the vocab '''
        return set(self.vocab)

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, char in ngrams(self.c, text):
            if context not in self.ngrams:
                self.ngrams[context] = {}
            self.ngrams[context][char] = self.ngrams[context].get(char, 0) + 1
            if char not in self.vocab:
                self.vocab.append(char)
        self.V = len(self.vocab)
        self.vocab.sort()
    
        
    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.ngrams:
            return 1/self.V
        denom = sum(self.ngrams[context].values()) + (self.V * self.k)
        if char not in self.ngrams[context]:
            return self.k/denom
        return (self.ngrams[context].get(char, 0) + self.k)/denom

    def random_char(self, context):
        ''' Returns a random char based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        prob_sum = 0
        for char in self.vocab:
            prob_sum += self.prob(context, char)
            if r < prob_sum:
                return char

    def random_text(self, length):
        ''' Returns text of the specified char length based on the
            n-grams learned by this model '''
        if self.c == 0:
            context = ""
        else:
            context = start_pad(self.c)
            
        text = ""
        for i in range(length):
            char = self.random_char(context[i:])
            text += char
            context += char
        
        return text
        

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        prob_log = 0
        for context, char in ngrams(self.c, text):
            prob_normal = self.prob(context, char)
            if prob_normal == 0:
                return float('inf')
            prob_log += math.log(prob_normal, 2)
        return 2**(-prob_log/len(text))
        

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c,k)
        self.lambdas = [1/(c+1) for _ in range(c+1)]
    
    def set_lambdas(self, new_lambdas):
        ''' Sets the interpolation weights to lambdas for each order of n-grams'''
        if len(new_lambdas) == len(self.lambdas) and sum(new_lambdas) == 1:
            self.lambdas = new_lambdas

    def get_vocab(self):
        ''' Returns the set of chars in the vocab '''
        return set(self.vocab)

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for i in range(self.c+1):
            for context, char in ngrams(i, text):
                if context not in self.ngrams:
                    self.ngrams[context] = {}
                self.ngrams[context][char] = self.ngrams[context].get(char, 0) + 1
                if char not in self.vocab:
                    self.vocab.append(char)
        
        self.V = len(self.vocab)
        self.vocab.sort()
    

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        prob = 0
        for i in range(self.c + 1):
            shortened_context = context[(self.c - i): ]
            if shortened_context in self.ngrams:
                denom = sum(self.ngrams[shortened_context].values()) + self.V * self.k
                prob += self.lambdas[i] * (self.ngrams[shortened_context].get(char,0) + self.k) / denom
            else:
                prob += self.lambdas[i] * (1 / self.V)
        return prob





################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.

def load_dataset(folder):
    '''Load training and evaluation dataset'''
    x = []
    y = []
    for code in COUNTRY_CODES:
        with open("./" + folder + "/" + code + ".txt", encoding='utf-8', errors='ignore') as input_file:
            for city in input_file:
                x.append(city.strip())
                y.append(code)
    return x, y

class CountriesModel:
    ''' A model for predicting the country (code) of origin of a city name '''
    def __init__(self, c, k, interpolation=False):
        models = {}
        for code in COUNTRY_CODES:
            if interpolation:
                models[code] = create_ngram_model(NgramModelWithInterpolation, "./train/" + code + ".txt", c, k)
                self.lambdas = [1/(c+1) for _ in range(c+1)]
            else:
                models[code] = create_ngram_model(NgramModel, "./train/" + code + ".txt", c, k)
        self.models = models
        self.c = c
    
    def set_lambdas(self, new_lambdas):
        ''' Sets the interpolation weights to lambdas for each order of n-grams'''
        if len(new_lambdas) == len(self.lambdas) and sum(new_lambdas) == 1:
            self.lambdas = new_lambdas
            for code in COUNTRY_CODES:
                self.models[code].set_lambdas(new_lambdas)
    
    def predict_country_lambda(self, city, new_lambdas):
        ''' Predicts the country (code) of origin of a city name '''
        self.set_lambdas(new_lambdas)
        return self.predict_country(city)

    def predict_country(self, city):
        ''' Predicts the country (code) of origin of a city name '''
        max_prob = 0
        predicted = ""
        for code in COUNTRY_CODES:
            c = self.models[code].c
            padded_city = start_pad(c) + str(city)
            prob = 1
            for i in range(len(city)):
                prob *= self.models[code].prob(padded_city[i:i+c], padded_city[i+c])
            if prob > max_prob:
                max_prob = prob
                predicted = code
        return predicted


def calculate_prediction(c,k, interpole):
    '''Calculate the prediction accuracy of the model'''
    x_test, y_test = load_dataset("val")
    model = CountriesModel(c, k, interpolation=interpole)
    count_True = 0
    for i in range(len(x_test)):
        if model.predict_country(x_test[i]) == y_test[i]:
            count_True += 1
    return count_True/len(x_test) 

def calculate_prediction_lambda(c,k,interpole, new_lambdas):
    '''Calculate the prediction accuracy of the model'''
    x_test, y_test = load_dataset("val")
    model = CountriesModel(c, k, interpolation=interpole)
    count_True = 0
    for i in range(len(x_test)):
        if model.predict_country_lambda(x_test[i], new_lambdas) == y_test[i]:
            count_True += 1
    return count_True/len(x_test)

def misclassified_cities(c,k,interpole):
    '''Get randomly 10 misclassified cities from the models'''
    x_test, y_test = load_dataset("val")
    model = CountriesModel(c, k, interpolation=interpole)
    misclassified = []
    mislabel_freq = {}
    for i in range(len(x_test)):
        predict = model.predict_country(x_test[i])
        if predict != y_test[i]:
            mislabel_freq[y_test[i]] = mislabel_freq.get(y_test[i], 0) + 1
            misclassified.append((x_test[i],y_test[i],predict))

    print(max(mislabel_freq, key=mislabel_freq.get)) #get the most mislabeled country from the models
    res = []
    for i in range(10):
        res.append(random.choice(misclassified))
    return res



if __name__ == '__main__':
    print(calculate_prediction_lambda(1,1,True,[0, 0.4, 0.3, 0.3]))
    print('################################')
    print(calculate_prediction_lambda(1,2,True,[0, 0, 0.5, 0.5]))
    print('################################')
    print(calculate_prediction_lambda(1,3,True,[0, 0, 0.6, 0.4]))
    print('################################')
    print(calculate_prediction_lambda(1,8,True,[0, 0.3, 0.4, 0.3]))
    print('################################')
    print(calculate_prediction_lambda(2,1,True,[0, 0.4, 0.3, 0.3]))
    print('################################')
    print(calculate_prediction_lambda(2,2,True,[0, 0, 0.5, 0.5]))
    print('################################')
    print(calculate_prediction_lambda(2,3,True,[0, 0, 0.6, 0.4]))
    print('################################')
    print(calculate_prediction_lambda(2,8,True,[0, 0.3, 0.4, 0.3]))
    print('################################')
    print(calculate_prediction_lambda(3,1,True,[0.25,0.25,0.25,0.25]))
    print('################################')
    print(calculate_prediction_lambda(3,2,True,[0.25,0.25,0.25,0.25]))
    print('################################')
    print(calculate_prediction_lambda(3,3,True,[0.25,0.25,0.25,0.25]))
    print('################################')
    print(calculate_prediction_lambda(3,8,True,[0.25,0.25,0.25,0.25]))
    print('################################')
    

    print(misclassified_cities(1,3,False))
    print('################################')
    print(misclassified_cities(3,2,True))
    print(calculate_prediction(3,1, True))
    print(calculate_prediction(1,8, False))

    print(calculate_prediction(0,0,False))
    print('################################')
    print(calculate_prediction(0,1,False))
    print('################################')
    print(calculate_prediction(0,2,False))
    print('################################')
    print(calculate_prediction(0,3,False))
    print('################################')
    print(calculate_prediction(0,8,False))
    print('################################')
    print(calculate_prediction(1,0,False))
    print('################################')
    print(calculate_prediction(1,1,True))
    print('################################')
    print(calculate_prediction(1,2,True))
    print('################################')
    print(calculate_prediction(1,3,True))
    print('################################')
    print(calculate_prediction(1,8,True))
    print('################################')
    print(calculate_prediction(2,0,False))
    print('################################')
    print(calculate_prediction(2,1,True))
    print('################################')
    print(calculate_prediction(2,2,True))
    print('################################')
    print(calculate_prediction(2,3,True))
    print('################################')
    print(calculate_prediction(2,8,True))
    print('################################')
    print(calculate_prediction(3,0,False))
    print('################################')
    print(calculate_prediction(3,1,True))
    print('################################')
    print(calculate_prediction(3,2,True))
    print('################################')
    print(calculate_prediction(3,3,True))
    print('################################')
    print(calculate_prediction(3,8,True))
    print('################################')
    print(calculate_prediction(4,0,False))
    print('################################')
    print(calculate_prediction(4,1,True))
    print('################################')
    print(calculate_prediction(4,2,True))
    print('################################')
    print(calculate_prediction(4,3,True))
    print('################################')
    print(calculate_prediction(4,4,True))
    print('################################')
    print(calculate_prediction(4,8,True))
    print('################################')
    print(calculate_prediction(7,0,False))
    print('################################')
    print(calculate_prediction(7,1,False))
    print('################################')
    print(calculate_prediction(7,2,False))
    print('################################')
    print(calculate_prediction(7,7,False))
    print('################################')
    print(calculate_prediction(7,8,False))
    print('################################')


    print(ngrams(1, 'abab'))
    print(ngrams(2, 'abc'))

    m = NgramModel(1,0)
    m.update('abab')
    print(m.get_vocab())
    m.update('abcd')
    print(m.get_vocab())
    print(m.prob('a', 'b'))
    print(m.prob('~', 'c'))
    print(m.prob('b', 'c'))

    m = NgramModel(0, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    res = [m.random_char('') for i in range(25)]
    print(res)
    print('Hello')
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.vocab)
    random.seed(42)
    print(m.random_text(25))

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print(m.random_text(250))

    print("#############################")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print(m.random_text(250))
    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print(m.random_text(250))
    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print(m.random_text(250))

    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 10)
    print(m.random_text(250))

    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 14)
    print(m.random_text(250))

    m = NgramModel(1,0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd'))
    print(m.perplexity('abca'))
    print(m.perplexity('abcda'))

    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('a', 'a'))
    print(m.prob('a', 'b'))
    print(m.prob('c', 'd'))
    print(m.prob('d', 'a'))

    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print(m.prob('a', 'a'))
    print(m.prob('a', 'b'))

    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('~a', 'b'))
    print(m.prob('ba', 'b'))
    print(m.prob('~c', 'd'))
    print(m.prob('bc', 'd'))

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2, k=7)
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2, k=7)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=7)
    m.set_lambdas([1/3] * 3)
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=7)
    m.set_lambdas([1/3] * 3)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, k=2)
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, k=2)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 7, k=2)
    m.set_lambdas([1/3] * 3)
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 7, k=2)
    m.set_lambdas([1/3] * 3)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([1/3] * 3)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=0.01)
    m.set_lambdas([1/4] * 4)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 4, k=0.01)
    m.set_lambdas([1/5] * 5)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([1/6] * 6)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 10, k=0.01)
    m.set_lambdas([1/11] * 11)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([1, 0, 0, 0, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 1, 0, 0, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 1, 0, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 0, 1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 0, 0, 1, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 0, 0, 0, 1])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.01, 0.01, 0.01, 0.01, 0.9, 0.06])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.05, 0.05, 0.05, 0.05, 0.75, 0.05])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.1, 0.1, 0.1, 0.1, 0.4, 0.2])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.1)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.2)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.5)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")
    
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=1)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=2)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=2)
    m.set_lambdas([0.25, 0.25, 0.25, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=1)
    m.set_lambdas([0.25, 0.25, 0.25, 0.25])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=1)
    m.set_lambdas([0.4, 0.3, 0.2, 0.1])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=1)
    m.set_lambdas([0.1, 0.2, 0.3, 0.4])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=1)
    m.set_lambdas([0.01, 0.01, 0.01, 0.97])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=1)
    m.set_lambdas([0.01, 0.01, 0.97, 0.01])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=1)
    m.set_lambdas([0.01, 0.97, 0.01, 0.01])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.97, 0.01, 0.01, 0.01])
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.25, 0.25, 0.25, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.4, 0.3, 0.2, 0.1])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.1, 0.2, 0.3, 0.4])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.01, 0.01, 0.01, 0.97])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.01, 0.01, 0.97, 0.01])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.01, 0.97, 0.01, 0.01])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=2)
    m.set_lambdas([0.97, 0.01, 0.01, 0.01])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
    
    print("#############################")
    
    