#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:25:42 2018

@author: zhouyuxufan
"""
from collections import Counter
from math import log

def p_c(p,p_ws,feature,V):
    p_logsum = 0
    for word,count in feature.items():
         p_w = p_ws.get(word,1/V)
         p_logsum = p_logsum+log(p_w)
    return p_logsum+log(p) 

class classifier:
    def __init__(self,p_ham,p_spam,p_w_ham,p_w_spam,V):
        self.p_ham= p_ham
        self.p_spam= p_spam
        self.p_w_ham = p_w_ham
        self.p_w_spam = p_w_spam
        self.V = V
        self.classified = []
        
        #maximum a posteriori
    def classify(self,features):                       
        classified = []
        for feature in features:
            if p_c(self.p_ham,self.p_w_ham,feature,self.V) > p_c(self.p_spam,self.p_w_spam,feature,self.V):
                classified.append((feature,p_c(self.p_ham,self.p_w_ham,feature,self.V),'ham'))
            else:
                classified.append((feature,p_c(self.p_spam,self.p_w_spam,feature,self.V),'spam'))
        self.classified = classified
        
    def evaluate(self,test_features):
        features = []
        labels = []
        for feature, label in test_features:
            features.append(feature)
            labels.append(label)
            cor = 0
            wro = 0
        self.classify(features)    
        for a,b in zip(labels,self.classified):
            if a == b[2]:
                cor = cor + 1
            else:
                wro = wro + 1
        accuracy = cor / (cor + wro)
        print(f'Accuracy is {accuracy}')  
                
    #maximum likelihood estimate
    @classmethod
    def train(cls,train_features,alpha=0.7):    
        n_spam = 0
        n_ham = 0
        dic_spam = {}
        dic_ham = {}
        for features,label in train_features:
            if label == 'spam':
                n_spam += 1
                for feature,count in features.items():
                    if feature in dic_spam:
                        dic_spam[feature] += count
                    else:
                        dic_spam[feature] = count
                        
            if label == 'ham':
                n_ham += 1
                for feature,count in features.items():
                    if feature in dic_ham:
                        dic_ham[feature] += count
                    else:
                        dic_ham[feature] = count 
                        
        z = dict(Counter(dic_ham)+Counter(dic_spam))
        V = len(z)
        counts_all = sum(dic_spam.values())+sum(dic_ham.values())
        #Laplace smoothing
        def lap(count):
            return (count+alpha)/(counts_all+alpha*V)
       
        p_ham = n_ham/(n_ham + n_spam)
        p_spam = 1- p_ham
        p_w_ham = {word:lap(count) for word, count in dic_ham.items()}
        p_w_spam = {word:lap(count) for word, count in dic_spam.items()}
        
        return cls(p_ham,p_spam,p_w_ham,p_w_spam,V)

def main():
    pass

if __name__ == '__main__':
    main()
