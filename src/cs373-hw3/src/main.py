# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

from classifier import BinaryClassifier
from perceptron import Perceptron, AveragedPerceptron
from naive_bayes import NaiveBayes
from utils import read_data
import utils
import os
from os import walk
import io
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import frequency_Counter
import token_to_vocab
from token_to_vocab import buildVocab

from config import args

if __name__ == '__main__':
    #filepath = '../data/given/'
    #build_vocab(filepath, vocab_size=args.vocab_size)
    legitimates = []
    phishings = []
    #fc = frequency_Counter()
    buildVocab()
    #"C:\\Users\\Tristan\\PycharmProjects\\test\\src\\cs373-hw3\\src\\vocab.txt"
    for a,b,f in walk("/home/tristan/PycharmProjects/phishbot/src/cs373-hw3/data/legitimate_htmls"):
        for file in f:
            legitimates.append("/home/tristan/PycharmProjects/phishbot/src/cs373-hw3/data/legitimate_htmls/"+file)
    for a, b, f in walk("/home/tristan/PycharmProjects/phishbot/src/cs373-hw3/data/phishing_htmls"):
        for file in f:
            phishings.append("/home/tristan/PycharmProjects/phishbot/src/cs373-hw3/data/phishing_htmls/" + file)
    LTrain, LTest = train_test_split(legitimates, test_size=0.20)
    PTrain, PTest = train_test_split(phishings, test_size=0.20)
    #train_data, test_data = read_data(filepath)
    #combinedTrain = LTrain.append(PTrain)
    test = list(LTest)
    test.extend(PTest)
    testLabels = [1]*len(LTest)
    testLabels.extend([-1] * len(PTest))
    combinedTest = (test, testLabels)
    #combinedTest = LTest
    #combinedTest.extend(PTest)
    """
    perc_classifier = Perceptron(args)
    perc_classifier.fit(train_data)
    acc, prec, rec, f1 = perc_classifier.evaluate(test_data)
    print('Perceptron Results:')
    print('Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f'%(acc, prec, rec, f1))
    
    avg_perc_classifier = AveragedPerceptron(args)
    avg_perc_classifier.fit(train_data)
    acc, prec, rec, f1 = avg_perc_classifier.evaluate(test_data)
    print('\nAveraged Perceptron Results:')
    print('Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f'%(acc, prec, rec, f1))
    """
    nb_classifier = NaiveBayes(args)
    if args.saved_model_legit == None and args.saved_model_phish == None:
        nb_classifier.fit(LTrain, PTrain)
    acc, prec, rec, f1, fpr = nb_classifier.evaluate(combinedTest, args.saved_model_legit, args.saved_model_phish)
    print('\nNaive Bayes Performance:')
    print('Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f, False Positive Rate: %.2f'%(acc, prec, rec, f1, fpr))

    """
    plt.plot([100, 500, 1000, 5000, 10000, 20000], [53.16, 61.46, 66.45, 75.42, 79.73, 79.73])
    plt.plot([100, 500, 1000, 5000, 10000, 20000], [54.15, 73.42, 75.75, 76.41, 79.40, 66.78])
    plt.legend(["Naive Bayes", "Averaged Perceptron"])
    plt.ylabel('Accuracy')
    plt.xlabel('Vocabulary Size')
    plt.title('Accuracy VS. Vocabulary Size')
    plt.show()
    """


