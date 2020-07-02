
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

from classifier import BinaryClassifier
import numpy as np
import utils
import frequency_Counter
#from frequencyCounter import getFeatureVector

class NaiveBayes(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.featDim = args.f_dim
        self.vocabSize = args.vocab_size
        #self.wordProbs = list()
        self.posProb = 0
        self.negProb = 0
        self.posWordCount = np.zeros(self.featDim)
        self.negWordCount = np.zeros(self.featDim)
        
    def fit(self, LTrain, PTrain):
        #TO DO: Learn the parameters from the training data
        #shuffle = np.arange(len(train_data[0]));
        #np.random.shuffle(shuffle)
        #featVectors = [[0]*300] *(len(LTrain)+len(PTrain))
        index = 0
        legitCount = 0
        fc = frequency_Counter.frequencyCounter()
        Legitmodel = np.empty([len(LTrain), self.vocabSize])
        for file in LTrain:
            print("\nFitting file ", index)
            example = open(file, "r", encoding = "utf8", errors="ignore")
            contents = example.read()
            encodedContents = contents.encode()
            #fc = frequency_Counter.frequencyCounter()
            fv = np.array(list(fc.getFeatureVector(contents)))

            Legitmodel[legitCount] = fv
            #featVectors[index] = fv
            #self.posWordCount += fv
            index+=1
            legitCount+=1
            example.close()
        Phishmodel = np.empty([len(PTrain), self.vocabSize])
        phishCount = 0
        for file in PTrain:
            print("\nFitting file ", index)
            example = open(file, "r", encoding = "utf8", errors = "ignore")
            contents = example.read()
            encodedContents = contents.encode()
            #*fc = frequency_Counter.frequencyCounter()
            fv = np.array(list(fc.getFeatureVector(contents)))
            Phishmodel[phishCount] = fv
            #np.append(Phishmodel, fv, axis=0)
            # featVectors[index] = fv
            #self.negWordCount += fv
            index += 1
            phishCount+=1
            example.close()

        np.savetxt('savedModelLegit.txt', Legitmodel, fmt="%d")
        np.savetxt('savedModelPhish.txt', Phishmodel, fmt="%d")

        #self.posProb = len(LTrain)/(len(LTrain)+len(PTrain))
        #self.negProb = len(PTrain)/(len(LTrain)+len(PTrain))

        """
        for i in range(0, len(train_data[1])):
            if train_data[1][i] == 1:
                self.posWordCount += fv[i]
            elif train_data[1][i] == -1:
                self.negWordCount += fv[i]
        """



        
    def predict(self, test_x, legitM='savedModelLegit.txt', phishM='savedModelPhish.txt'):
        #TO DO: Compute and return the output for the given test inputs
        labels = []
        #fv = np.array(utils.get_feature_vectors(test_x, False))
        print("Loading saved LEGIT model...")
        legit = np.loadtxt(legitM)
        print("Loading saved PHISH model...")
        phish = np.loadtxt(phishM)
        print("total files to classify: ", len(test_x))
        index = 0

        self.posWordCount = np.sum(legit, axis=0)
        self.negWordCount = np.sum(phish, axis=0)
        self.posProb = len(legit)/(len(legit)+len(phish))
        self.negProb = len(phish)/(len(legit)+len(phish))

        posCount = np.sum(self.posWordCount)
        negCount = np.sum(self.negWordCount)

        for file in test_x:
            example = open(file, "r", encoding="utf8", errors="ignore")
            contents = example.read()
            print("classifying file ", index)
            fc = frequency_Counter.frequencyCounter()
            fv = np.array(list(fc.getFeatureVector(contents)))
            probPos = np.math.log(self.posProb)
            probNeg = np.math.log(self.negProb)
            for j in range(self.vocabSize):
                word_freq = fv[j]
                if word_freq == 0:
                    continue
                posWordOccur = self.posWordCount[j] + 1
                negWordOccur = self.negWordCount[j] + 1
                probPos += np.math.log(np.longdouble(word_freq*posWordOccur/(posCount+self.vocabSize)))
                probNeg += np.math.log(np.longdouble(word_freq*negWordOccur/(negCount+self.vocabSize)))

            total = probPos + probNeg
            if probPos > (probNeg+100):
                labels.append(1)
            else:
                labels.append(-1)
            index+=1
            example.close()

        return labels
