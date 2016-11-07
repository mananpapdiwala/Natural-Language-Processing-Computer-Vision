###################################
# CS B551 Fall 2016, Assignment #3
#
# Names and user ids:
# Anurag Jain: jainanur
# Pratik Jain: jainps
# Manan Papdiwala: mampapdi
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Report:
#
# Approach:
# For HMM we have created a table called viterbi tables which stores all the values generated during the viterbi
# algorithm. Values in one column help us generate values in succeeding column. This table also helps us trace back
# the final solution.
# In Complex we have created another table for storing transitions such as P(si+2|si). The values of this table and the
# P(si+1|si) collectively help us to find the final probabilities
#
# Training data used: bc.train
# Testing data used: bc.test
# Accuracies:
# ==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       90.59%               33.65%
#            2. HMM:       95.06%               54.85%
#        3. Complex:       69.83%               22.85%
# ----
# Training data used: bc.train
# Testing data used: bc.test.tiny
# Accuracies:
# ==> So far scored 3 sentences with 42 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       92.86%               33.33%
#            2. HMM:       97.62%               66.67%
#        3. Complex:       76.19%                0.00%
# ----
####

import math
import copy


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#


class Solver:
    # Initialize class variables
    def __init__(self):
        self.POS = ["det", ".", "x", "noun", "verb", "prt", "pron", "num", "adp", "adv", "adj",
                    "conj"]  # List of possible parts of speech
        self.total_words = 0  # Total no. of words
        self.s1 = {}  # initial probabilities of each POS
        self.transition_probabilities = {}  # P(Si+1|si)
        self.emission_probabilities = {}  # P(Wi|Si)
        self.count_for_each_part_of_speech = {}  # count of occurrences of each POS
        self.emission_count = {}  #
        self.secondTransitionProbabilities = {}  # P(Si+2|si)

    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling
    def posterior(self, sentence, label):
        x = 0
        for a in label:
            if a != 0:
                x += math.log(a)

        return x

    # Do the training!
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
                    if "end" in s[sentence[1][i]]:  # Check if sentence ends in a word
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

        s2 = {}
        for partsOfSpeech in self.POS:
            s2[partsOfSpeech] = {}
        for sentence in data:
            for i in range(2, len(sentence[1])):
                if sentence[1][i] in s2[sentence[1][i - 2]]:
                    s2[sentence[1][i - 2]][sentence[1][i]] += 1
                else:
                    s2[sentence[1][i - 2]].update({sentence[1][i]: 1})
        for key in s2.keys():
            for innerKey in s2.keys():
                if innerKey not in s2[key]:
                    s2[key].update({innerKey: 1})
        for key in s2.keys():
            key_sum = sum(s2[key].values())
            for innerKey in s2.keys():
                s2[key][innerKey] = (s2[key][innerKey] * 1.0) / key_sum
        self.secondTransitionProbabilities = s2

    # Calculate probability of POS given word
    def get_simplified_probability(self, word):
        s_w = []
        for parts_of_speech in self.POS:
            w_s = self.emission_probabilities[parts_of_speech][word] if word in self.emission_probabilities[
                parts_of_speech] else 1.0 / self.count_for_each_part_of_speech[parts_of_speech]
            s = (self.count_for_each_part_of_speech[parts_of_speech] * 1.0) / self.total_words
            w = 0
            for pos in self.POS:
                w += (self.emission_probabilities[pos][word] if word in self.emission_probabilities[pos] else 1.0 /
                                                                                                              self.count_for_each_part_of_speech[
                                                                                                                  pos]) * (
                         (self.count_for_each_part_of_speech[pos] * 1.0) / self.total_words)
            s_w.append([parts_of_speech, (w_s * s) / w])
        max_row = 0
        max_probability = s_w[0][1]
        for row in range(0, len(s_w)):
            if s_w[row][1] > max_probability:
                max_probability = s_w[row][1]
                max_row = row
        return s_w[max_row]

    # P(si/si+1)
    def get_transition(self, transition, from_tag, to_tag):
        return transition * (self.count_for_each_part_of_speech[from_tag]) / self.count_for_each_part_of_speech[to_tag]

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        result = []
        marginal_probability = []
        for i in range(0, len(sentence)):
            pos, prob = self.get_simplified_probability(sentence[i])
            result.append(pos)
            marginal_probability.append(prob)
        return [[result], [marginal_probability]]

    def hmm(self, sentence):
        viterbi_table = {}
        for partsOfSpeech in self.POS:
            viterbi_table[partsOfSpeech] = {}
        for i in range(len(sentence)):
            for partsOfSpeech in self.POS:
                if i != 0:
                    previous_max = max(viterbi_table, key=lambda x: (
                        viterbi_table[x][sentence[i - 1]] + math.log(self.transition_probabilities[x][partsOfSpeech])))
                if sentence[i] not in self.emission_probabilities[partsOfSpeech]:
                    # ep = 10 ** (-6)
                    ep = 1.0 / self.total_words
                else:
                    ep = self.emission_probabilities[partsOfSpeech][sentence[i]]
                if i == 0:
                    viterbi_table[partsOfSpeech][sentence[i]] = math.log(ep) + math.log(self.s1[partsOfSpeech])
                else:
                    viterbi_table[partsOfSpeech][sentence[i]] = math.log(ep) + math.log(
                        self.get_transition(self.transition_probabilities[previous_max][partsOfSpeech], previous_max,
                                            partsOfSpeech)) + viterbi_table[previous_max][sentence[i - 1]]

        result = []
        marginal_probability = []
        for i in range(len(sentence) - 1, -1, -1):
            if i == len(sentence) - 1:
                current_max = max(viterbi_table, key=lambda x: viterbi_table[x][sentence[i]])
            else:
                current_max = previous_max
            result.append(current_max)
            baseSum = sum([math.exp(viterbi_table[key][sentence[i]]) for key in viterbi_table.keys()])
            marginal_probability.append(math.exp(viterbi_table[current_max][sentence[i]]) / baseSum)

            if i != 0:
                previous_max = max(viterbi_table, key=lambda x: (
                    viterbi_table[x][sentence[i - 1]] + math.log(
                        self.get_transition(self.transition_probabilities[x][current_max], x, current_max))))
        result.reverse()
        marginal_probability.reverse()
        #for i in range(len(marginal_probability)):
        #    marginal_probability[i] = math.exp(marginal_probability[i])
        return [[result], [marginal_probability]]

    def complex(self, sentence):
        # A 2-D array to store the probability distribution for each word
        # Of the form { noun : { 0: probability, 1: probability }, verb : { 0: probability, 1: probability } }

        variable_elimination_result = {}
        for partsOfSpeech in self.POS:
            variable_elimination_result[partsOfSpeech] = {}
        for i in range(0, len(sentence)):
            for partsOfSpeech in self.POS:
                if i == 0:
                    # emission = self.emission_probabilities[partsOfSpeech][sentence[i]] if sentence[i] in \
                    #    self.emission_probabilities[partsOfSpeech] else 1.0 / self.count_for_each_part_of_speech[
                    #        partsOfSpeech]
                    emission = self.emission_probabilities[partsOfSpeech][sentence[i]] if sentence[i] in \
                                                                                          self.emission_probabilities[
                                                                                              partsOfSpeech] else (
                    10 ** (-6))
                    s = (self.s1[partsOfSpeech] * 1.0) * emission
                    variable_elimination_result[partsOfSpeech].update({i: math.log(s)})

                elif i == 1:
                    p = 0
                    if sentence[i] not in self.emission_probabilities[partsOfSpeech]:
                        ep = 10 ** (-6)
                        # ep = 1.0 / self.count_for_each_part_of_speech[partsOfSpeech]
                    else:
                        ep = self.emission_probabilities[partsOfSpeech][sentence[i]]
                    for pos in self.POS:
                        p += (self.get_transition(self.transition_probabilities[pos][partsOfSpeech], pos,
                                                  partsOfSpeech) * math.exp(
                            variable_elimination_result[pos][i - 1]))
                    p = math.log(p) + math.log(ep)
                    variable_elimination_result[partsOfSpeech].update({i: p})

                else:
                    p = 0
                    if sentence[i] not in self.emission_probabilities[partsOfSpeech]:
                        ep = 10 ** (-6)
                        # ep = 1.0 / self.count_for_each_part_of_speech[partsOfSpeech]
                    else:
                        ep = self.emission_probabilities[partsOfSpeech][sentence[i]]
                    for j in range(i - 2, i):
                        for pos in self.POS:
                            if j == i - 2:
                                p += math.log(
                                    self.get_transition(self.secondTransitionProbabilities[pos][partsOfSpeech], pos,
                                                        partsOfSpeech))
                            else:
                                p += math.log(
                                    self.get_transition(self.secondTransitionProbabilities[pos][partsOfSpeech], pos,
                                                        partsOfSpeech)) + variable_elimination_result[pos][j]
                        p += math.log(ep)
                    variable_elimination_result[partsOfSpeech].update({i: p})

        result = []
        marginal_probability = []
        for i in range(0, len(sentence)):
            max_p_o_s = variable_elimination_result.keys()[0]
            max_probability = variable_elimination_result[max_p_o_s][i]
            for key in variable_elimination_result.keys():
                if (variable_elimination_result[key][i]) > max_probability:
                    max_p_o_s = key
                    max_probability = variable_elimination_result[key][i]
            result.append(max_p_o_s)
            marginal_probability.append(max_probability)
        for i in range(len(marginal_probability)):
            marginal_probability[i] = math.exp(marginal_probability[i])
        return [[result], [marginal_probability]]

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
