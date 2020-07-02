#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

from classifier import BinaryClassifier
import numpy as np
import utils



class Perceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.bias = 0
        self.featDim = args.f_dim
        self.vocabSize = args.vocab_size
        self.numIter = args.num_iter
        self.learnRate = args.lr
        self.binaryFeats = args.bin_feats
        self.weights = np.zeros(self.featDim);

        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        shuffle = np.arange(len(train_data[0]));
        np.random.shuffle(shuffle)
        fv = np.array(utils.get_feature_vectors(train_data[0],False))
        for h in range(self.numIter):
            for i in shuffle:
                predictedLabel = np.dot(fv[i], self.weights) + self.bias;
                if(predictedLabel > 0):
                    result = 1
                else:
                    result = -1

                if (result != train_data[1][i]):
                    self.weights += self.learnRate*train_data[1][i] *fv[i]
                    self.bias += self.learnRate * train_data[1][i];

        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        labels = []
        fv = np.array(utils.get_feature_vectors(test_x, False))
        for i in range((len(test_x))):
            predictedLabel = np.dot(fv[i], self.weights) + self.bias;
            if (predictedLabel > 0):
                labels.append(1)
            else:
                labels.append(-1)

        return labels



class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.bias = 0
        self.featDim = args.f_dim
        self.vocabSize = args.vocab_size
        self.numIter = args.num_iter
        self.learnRate = args.lr
        self.binaryFeats = args.bin_feats
        self.weights = np.zeros(self.featDim);
        self.survival = 1
        self.tempWeights = self.weights
                
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        shuffle = np.arange(len(train_data[0]));
        np.random.shuffle(shuffle)
        fv = np.array(utils.get_feature_vectors(train_data[0], False))
        for h in range(self.numIter):
            for i in shuffle:
                predictedLabel = np.dot(fv[i], self.weights) + self.bias;
                if (predictedLabel > 0):
                    result = 1
                else:
                    result = -1

                if (result != train_data[1][i]):
                    tempWeights = self.weights
                    self.weights = tempWeights + self.learnRate * train_data[1][i] * fv[i]
                    self.bias += self.learnRate * train_data[1][i];
                    self.tempWeights = ((self.survival*self.tempWeights) + self.weights)/(self.survival + 1)
                    self.survival = 1
                else:
                    self.survival += 1
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        labels = []
        fv = np.array(utils.get_feature_vectors(test_x, False))
        for i in range((len(test_x))):
            predictedLabel = np.dot(fv[i], self.weights) + self.bias;
            if (predictedLabel > 0):
                labels.append(1)
            else:
                labels.append(-1)

        return labels

