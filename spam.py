#!/usr/bin/env python
"""Spam mail classifier."""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import collections


def percent(num, total):
    return 100 * float(num) / total


df = pd.read_csv('spambase.data', names=[i for i in range(58)])

train, test = train_test_split(df, test_size=0.2)

train_counts = (train.loc[:, [i for i in range(57)]]).values
train_targets = np.array(train[57])

classifier = MultinomialNB()
classifier.fit(train_counts, train_targets)


test_counts = test.loc[:, [i for i in range(57)]]

predictions = classifier.predict(test_counts)

count_pred = collections.Counter(predictions)

total = sum(count_pred.values())
non_spam_pct = percent(count_pred[0], total)
spam_pct = percent(count_pred[1], total)

print "Test total %d" % total
print "Non Spam %d" % count_pred[0]
print "Spam     %d\n" % count_pred[1]

print "Percentages"
print "Non Spam: %.2f%s\nSpam:     %.2f%s" % (non_spam_pct, '%', spam_pct, '%')
