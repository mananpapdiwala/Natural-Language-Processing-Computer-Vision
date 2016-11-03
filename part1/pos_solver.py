###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        # Calculating P(S1)
        POS = ["det", ".", "x", "noun", "verb", "prt", "pron", "num", "adp", "adv", "pron", "adj", "conj"]
        s1 = {}
        for entry in POS:
            s1.update({entry: 0})
        for sentence in data:
            s1[sentence[1][0]] += 1
        for key in s1.keys():
            if s1[key] == 0:
                s1[key] = 1                 # if no sentence starts with a Parts of Speech
        size_of_data = sum(s1.values())
        for key in s1.keys():
            s1[key] = (s1[key] * 1.0) / (size_of_data * 1.0)
        # Calculating P(Si+1|si)
        s = {}
        for key in s1.keys():
            s[key] = {}
        for sentence in data:
            for i in range(1,len(sentence[1])):
                if sentence[1][i] in s[sentence[1][i-1]]:
                    s[sentence[1][i - 1]][sentence[1][i]] += 1
                else:
                    s[sentence[1][i - 1]].update({sentence[1][i]:1})
        for key in s.keys():
            for inner_key in s.keys():
                if inner_key not in s[key]:
                    s[key].update({inner_key:1})
        for key in s.keys():
            key_sum = sum(s[key].values())
            for inner_key in s.keys():
                s[key][inner_key] = (s[key][inner_key]*1.0) / (key_sum*1.0)
        # Calculating P(Wi|Si)
        '''w_s = {}
        for sentence in data:
            for i in range(1, len(sentence[1])):
                if sentence[1][i] in w_s[sentence[1][i - 1]]:
                    w_s[sentence[1][i - 1]][sentence[1][i]] += 1
                else:
                    w_s[sentence[1][i - 1]].update({sentence[1][i]: 1})
        for key in w_s.keys():
            for inner_key in w_s.keys():
                if inner_key not in w_s[key]:
                    w_s[key].update({inner_key: 1})
        for key in w_s.keys():
            key_sum = sum(w_s[key].values())
            for inner_key in w_s.keys():
                w_s[key][inner_key] = (w_s[key][inner_key] * 1.0) / (key_sum * 1.0)'''
        pass

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

    def hmm(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [] ]

    def complex(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

