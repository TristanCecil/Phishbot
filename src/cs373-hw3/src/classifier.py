#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

class BinaryClassifier(object):
    
    def __init__(self):
        raise NotImplementedError
    
    def fit(self, train_data):
        raise NotImplementedError
    
    def predict(self, test_x):
        raise NotImplementedError
    
    def evaluate(self, test_data, legitM, phishM):
        test_x, test_y = test_data
        pred_y = self.predict(test_x, legitM, phishM)
        tp, tn, fp, fn = 0., 0., 0., 0.
        for py, gy in zip(pred_y, test_y):
            if py == -1 and gy == -1:
                tn += 1
            elif py == -1 and gy == 1:
                fn += 1
            elif py == 1 and gy == 1:
                tp += 1
            elif py == 1 and gy == -1:
                fp += 1
        cm = ((tn, fn), (fp, tp))
        return self.metrics(cm)
    
    def metrics(self, confusion_matrix):
        true_positives = confusion_matrix[1][1]
        false_positives = confusion_matrix[1][0]
        false_negatives = confusion_matrix[0][1]
        true_negatives = confusion_matrix[0][0]
        total_size = true_positives + true_negatives + false_positives + false_negatives
        acc = 100 * (true_positives + true_negatives) / (total_size * 1.0)
        prec = 100 *  (true_positives * 1.0) / (true_positives + false_positives + 0.01)
        rec = 100 * (true_positives * 1.0) / (true_positives + false_negatives + 0.01)
        fpr = 100 * (false_positives * 1.0) / (false_positives + true_negatives + 0.01)
        if prec == 0 and rec == 0:
            f1 = 0
        else:
            f1 = (2.0 * prec * rec) / (prec + rec)
        return (acc, prec, rec, f1, fpr)

