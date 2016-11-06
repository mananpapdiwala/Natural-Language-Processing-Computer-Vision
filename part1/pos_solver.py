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

import math
import copy


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
import sys


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def __init__(self):
        # Initialising the parts of speech dictionary
        self.POS = ["det", ".", "x", "noun", "verb", "prt", "pron", "num", "adp", "adv", "adj", "conj"]
        self.total_words = 0
        self.s1 = {}
        self.transition_probabilities = {}
        self.emission_probabilities = {}
        self.count_for_each_part_of_speech = {}
        self.emission_count = {}

    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        # Calculating P(s1)

        s1 = {}
        for partsOfSpeech in self.POS:
            s1.update({partsOfSpeech: 0})

        # Counting the number of times a particular part of speech starts with a sentence
        for sentence in data:
            s1[sentence[1][0]] += 1

        # Check if sentence does not start from a part of speech and set it to 1
        for key in s1.keys():
            if s1[key] == 0:
                s1[key] = 1.0

        # Counting the probability for each part of speech to start in a sentence
        size_of_data = sum(s1.values())
        for key in s1.keys():
            s1[key] = (s1[key] * 1.0) / (size_of_data * 1.0)

        self.s1 = s1

        # Calculating P(Si+1|si)
        s = {}
        for key in self.s1.keys():
            s[key] = {}
        for sentence in data:
            for i in range(1, len(sentence[1])):
                if i != len(sentence[1]) - 1:
                    if sentence[1][i] in s[sentence[1][i - 1]]:
                        s[sentence[1][i - 1]][sentence[1][i]] += 1
                    else:
                        s[sentence[1][i - 1]].update({sentence[1][i]: 1})
                else:
                    if "end" in s[sentence[1][i]]:
                        s[sentence[1][i]]["end"] += 1
                    else:
                        s[sentence[1][i]].update({"end": 1})
        for key in s.keys():
            for inner_key in s.keys():
                if inner_key not in s[key]:
                    s[key].update({inner_key: 1})
        for key in s.keys():
            if "end" not in s[key]:
                s[key].update({"end": 1})
        for key in s.keys():
            key_sum = sum(s[key].values())
            for inner_key in s[key]:
                s[key][inner_key] = (s[key][inner_key] * 1.0) / key_sum
        self.transition_probabilities = s

        # Calculating P(Wi|Si)
        w_s = {}
        for partsOfSpeech in self.POS:
            w_s[partsOfSpeech] = {}

        for sentence in data:
            for i in range(0, len(sentence[0])):
                if sentence[0][i] in w_s[sentence[1][i]]:
                    w_s[sentence[1][i]][sentence[0][i]] += 1
                else:
                    w_s[sentence[1][i]].update({sentence[0][i]: 1})
        self.emission_count = copy.deepcopy(w_s)
        for key in w_s.keys():
            key_sum = sum(w_s[key].values())
            for inner_key in w_s[key].keys():
                w_s[key][inner_key] = (w_s[key][inner_key] * 1.0) / key_sum

        self.emission_probabilities = w_s
        for partsOfSpeech in self.POS:
            temp_sum = sum(self.emission_count[partsOfSpeech].values())
            self.count_for_each_part_of_speech[partsOfSpeech] = 1 if temp_sum == 0 else temp_sum

        self.total_words = sum([sum(self.emission_count[row].values()) for row in self.emission_count])

    def get_simplified_probability(self, word):
        s_w = []
        for parts_of_speech in self.POS:
            w_s = self.emission_probabilities[parts_of_speech][word] if word in self.emission_probabilities[parts_of_speech] else 1.0 / self.count_for_each_part_of_speech[parts_of_speech]
            s = (self.count_for_each_part_of_speech[parts_of_speech] * 1.0) / self.total_words
            w = 0
            for pos in self.POS:
                w += (self.emission_probabilities[pos][word] if word in self.emission_probabilities[pos] else 1.0 / self.count_for_each_part_of_speech[pos]) * ((self.count_for_each_part_of_speech[pos] * 1.0) / self.total_words)
            s_w.append([parts_of_speech, (w_s * s)/w])
        max_row = 0
        max_probability = s_w[0][1]
        for row in range(0, len(s_w)):
            if s_w[row][1] > max_probability:
                max_probability = s_w[row][1]
                max_row = row
        return s_w[max_row]

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        simplified_result = []
        marginal_probability = []
        for i in range(0, len(sentence)):
            pos, prob = self.get_simplified_probability(sentence[i])
            simplified_result.append(pos)
            marginal_probability.append(math.log10(prob))
        return [[simplified_result], [marginal_probability]]

    def hmm(self, sentence):
        viterbi_table = {}
        for partsOfSpeech in self.POS:
            viterbi_table[partsOfSpeech] = {}
        for i in range(len(sentence)):
            for partsOfSpeech in self.POS:
                if i != 0:
                    previousMax = max(viterbi_table, key=lambda x: (
                    viterbi_table[x][sentence[i - 1]] + math.log(self.transition_probabilities[x][partsOfSpeech])))
                if sentence[i] not in self.emission_probabilities[partsOfSpeech]:
                    ep = 1.0 / self.count_for_each_part_of_speech[partsOfSpeech]
                else:
                    ep = self.emission_probabilities[partsOfSpeech][sentence[i]]
                if i == 0:
                    viterbi_table[partsOfSpeech][sentence[i]] = math.log(ep) + math.log(self.s1[partsOfSpeech])
                else:
                    viterbi_table[partsOfSpeech][sentence[i]] = math.log(ep) + math.log(
                        self.transition_probabilities[previousMax][partsOfSpeech]) + \
                                                                viterbi_table[previousMax][sentence[i - 1]]

        l1 = []
        l2 = []
        for i in range(len(sentence) - 1, -1, -1):
            if i == len(sentence) - 1:
                currentMax = max(viterbi_table, key=lambda x: viterbi_table[x][sentence[i]])
            else:
                currentMax = previousMax
            l1.append(currentMax)
            l2.append(viterbi_table[currentMax][sentence[i]])

            if i != 0:
                previousMax = max(viterbi_table, key=lambda x: (
                viterbi_table[x][sentence[i - 1]] + math.log(self.transition_probabilities[x][currentMax])))
        l1.reverse()
        l2.reverse()
        return [[l1], [l2]]

    def complex(self, sentence):
        return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]

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
