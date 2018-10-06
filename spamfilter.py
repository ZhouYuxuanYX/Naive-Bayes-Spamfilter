"""
Usage:
  spamfilter train    -m <path> -c <path>
  spamfilter classify -m <path> <path>

Options:
  -c <path>, --corpuspath  Path to the corpus that should be used to train the
                           spamfilter. The path should contain two
                           directories, ham and spam, which contain the ham
                           and spam emails in individual files.
  -m <path>, --modelpath   Path to the model. In training mode this is where
                           the model will be written to, in classification
                           mode this is where the model will be read from.
"""
import os
import sys
import chardet
import random
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from NBclassifier import classifier
from os import path
import numpy as np
import _pickle as cPickle

stoplist = stopwords.words('english')

def FileRead(folder):
    List = []
    FileList = os.listdir(folder)
    for File in FileList:
        with open(folder + File,encoding="utf-8", errors="surrogateescape") as f:
            List.append(f.read())
    return List

def preprocess(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def get_features(text):
    return {word: count for word, count in Counter(preprocess(text)).items() \
    if not word in stoplist}                                               

def evaluate(set, classifier):
    print('Accuracy on the set ='
          + str(classify.accuracy(classifier,set)))
     
def main(opts):
    #load data
    if opts['train']:
        spam_train_ini = FileRead(opts['--corpuspath'] + '/spam/')
        ham_train_ini = FileRead(opts['--corpuspath'] + '/ham/')
        spam_train = [(email,'spam') for email in spam_train_ini]
        ham_train =[(email,'ham') for email in ham_train_ini]
        emails_train = ham_train + spam_train
        print(f'Training size is: {len(emails_train)}')
        random.shuffle(emails_train)
    
        #extract the features
        train_features = [(get_features(email), label) for (email, label)
        in emails_train]
    
        #train a classifier
        clas = classifier.train(train_features)
        clas.evaluate(train_features)
#        classifier.show_most_informative_features(20)
        
        # save the classifier
        if not os.path.exists(opts['--modelpath']):
            os.makedirs(opts['--modelpath'])
        with open(opts['--modelpath']+ '/NBclassifier.pkl', 'wb+') as fid:
            cPickle.dump(clas, fid)  
        
    
    
    if opts['classify']:
        spam_test_ini = FileRead(opts['<path>'] + '/spam/')
        ham_test_ini = FileRead(opts['<path>'] + '/ham/')
        spam_test = [(email,'spam') for email in spam_test_ini] 
        ham_test = [(email,'ham') for email in ham_test_ini]
        emails_test= ham_test + spam_test
        print(f'Test size is: {len(emails_test)}')
        random.shuffle(emails_test)
        test_features = [(get_features(email), label) for (email, label)
        in emails_test]
    
        # load it again
        with open(opts['--modelpath']+ '/NBclassifier.pkl', 'rb') as fid:
            clas = cPickle.load(fid)
        
        clas.evaluate(test_features)
#        classifier.show_most_informative_features(20)

    
if __name__ == '__main__':
    import docopt
    opts = docopt.docopt(__doc__)
    print(opts)
    main(opts)

