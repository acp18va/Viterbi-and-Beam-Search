# ======================================================================================================================
# Importing
# ======================================================================================================================

from collections import Counter
import sys
import numpy as np
import time
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import heapq

# Setting random seed
random.seed(11242)

# declaring some global variables
depochs = 5
feat_red = 0
learn = []

# printing number of iterations and default feature reduction
print("\nDefault no. of epochs: ", depochs)
print("\nDefault feature reduction threshold: ", feat_red)


# ======================================================================================================================
# Taking file input and converting it to list of tuples
# ======================================================================================================================

print("\nLoading the data \n")


def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


# setting the command line arguments for training and test set
# calling the load_dataset_sents function to load training and test data
train_data = load_dataset_sents(sys.argv[2])
test_data = load_dataset_sents(sys.argv[3])

# unique tags as labels
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

# ======================================================================================================================
# function to define the feature set
# ======================================================================================================================

print("\nDefining the feature space \n")


# feature space of cw_ct
def cw_ct_counts(data, freq_thresh=5):  # data input as (cur_word, cur_tag)

    cw_c1_c = Counter()
    # counting the tuples
    for doc in data:
        cw_c1_c.update(Counter(doc))
    # returning the counted feature space
    return Counter({k: v for k, v in cw_c1_c.items() if v > freq_thresh})


# ======================================================================================================================
# function phi_1 to check if a tuple is present in feature space
# ======================================================================================================================

# feature representation of a tuple in cw_ct_count
def phi_1(sent, cw_ct_count):  # sent as (cur_word, cur_tag)
    for item in sent:
        # if item in feature space returns 1 as a count else 0
        if item in cw_ct_count.keys():
            return 1
        else:
            return 0


# ======================================================================================================================
# Scoring function for viterbi search
# ======================================================================================================================

def scoring_viterbi(doc, cw_ct_count, weights):
    # unzipping words and tags
    sentence, tags = list(zip(*doc))
    # calling viterbi search for the maximum scoring sequence
    max_scoring_seq = viterbi(sentence, weights, cw_ct_count)
    # zipping it again
    sequence = [(k, v) for k, v in zip(sentence, max_scoring_seq)]
    # returning the maximum scoring sequence
    return sequence


# ======================================================================================================================
# Scoring function for beam search
# ======================================================================================================================

def scoring_beam(doc, cw_ct_count, weights):
    # unzipping words and tags
    sentence, tags = list(zip(*doc))
    # calling beam search for the maximum scoring sequence and specifying size of beam
    max_scoring_seq = beam_search(sentence, weights, cw_ct_count, beam=1)
    # zipping it again
    sequence = [(k, v) for k, v in zip(sentence, max_scoring_seq)]
    # returning the maximum scoring sequence
    return sequence


# ======================================================================================================================
# train function for structured perceptron
# ======================================================================================================================

def train_perceptron(data, cw_ct_count, epochs, shuffle=True, feature = 1):

    # variables used as metrics for performance and accuracy
    iterations = range(len(data) * epochs)
    false_prediction = 0
    false_predictions = []

    # initialising our weights dictionary as a counter
    # counter.update allows addition of relevant values for keys
    # a normal dictionary replaces the key-value pair
    weights = Counter()

    start = time.time()

    # multiple passes
    for epoch in range(epochs):
        false = 0
        now = time.time()

        # going through each sentence-tag_seq pair in training_data

        # shuffling if necessary
        if shuffle == True:
            random.shuffle(data)

        for doc in data:
            if feature == 1:
                # retrieve the highest scoring sequence using viterbi or beam
                max_scoring_seq = scoring_viterbi(doc, cw_ct_count, weights)
            else:
                max_scoring_seq = scoring_beam(doc, cw_ct_count, weights)

            # if the prediction is wrong
            if max_scoring_seq != doc:
                correct = Counter(doc)

                # negate the sign of predicted wrong
                predicted = Counter({k: -v for k, v in Counter(max_scoring_seq).items()})

                # add correct
                weights.update(correct)

                # negate false
                weights.update(predicted)

                """Recording false predictions"""
                false += 1
                false_prediction += 1
            false_predictions.append(false_prediction)

        print("Epoch: ", epoch + 1,
              " / Time for epoch: ", round(time.time() - now, 2),
              " / No. of false predictions: ", false)
        # appending learn with no. of errors in each iteration for plotting learning graph
        learn.append(false)
    return weights, learn


# ======================================================================================================================
# test function for structured perceptron
# ======================================================================================================================

# testing the learned weights
def test_perceptron(data, cw_ct_count,  weights, feature = 1):

    correct_tags = []
    predicted_tags = []

    i = 0

    for doc in data:
        _, tags = list(zip(*doc))

        correct_tags.extend(tags)

        if feature == 1:
            # retrieve the highest scoring sequence with viterbi or beam search
            max_scoring_seq = scoring_viterbi(doc, cw_ct_count, weights)
        else:
            max_scoring_seq = scoring_beam(doc, cw_ct_count, weights)

        _, pred_tags = list(zip(*max_scoring_seq))

        predicted_tags.extend(pred_tags)

    return correct_tags, predicted_tags


# ======================================================================================================================
# function to evaluate the results using f1_score
# ======================================================================================================================

def evaluate(correct_tags, predicted_tags):

    # calculating the f1_score
    f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=["PER", "LOC", "ORG", "MISC"])

    print("F1 Score: ", round(f1, 5))

    return f1


# ======================================================================================================================
# Function implementing beam search
# ======================================================================================================================

def beam_search(sentence, weights, cw_ct_count, beam=1):
    # declaring beam_s matrix to store the score of each tuple
    beam_s = np.zeros((len(all_tags), len(sentence)))
    # declaring back pointer matrix to store the index of maximum value for each tuple
    back = np.zeros((len(all_tags), len(sentence)))

    # calculating the beam_s score for first column
    for y in range(len(all_tags)):
        pair = [(sentence[0], all_tags[y])]
        # calling phi_1 for each tuple
        phi = phi_1(pair, cw_ct_count)
        # calculating and storing the score of the tuple
        beam_s[y][0] = phi * weights[(sentence[0], all_tags[y])]

    # calculating the beam_s score for rest of the columns
    for n in range(1, len(sentence)):
        # finding the top k maximum score for previous column and storing the indices in new_index
        new_index = heapq.nlargest(beam, range(len(beam_s[n-1])), beam_s[n-1].take)

        for y in range(len(all_tags)):
            pair = [(sentence[n], all_tags[y])]
            # calling phi_1 for each tuple
            phi = phi_1(pair, cw_ct_count)
            # calling get_max function to find maximum in top k indices
            max = get_max(new_index, beam_s[:, n-1])
            # calculating and storing the score of the tuple
            beam_s[y][n] = max + phi * weights[(sentence[n], all_tags[y])]
            # storing the index of maximum value in the back pointer matrix
            back[y][n] = np.argmax(beam_s[n - 1])
    # finding the list of indices of maximum values in each column
    index = np.argmax(beam_s, axis=0)
    # converting the list of indices to the respective tag sequence
    tag_seq = [all_tags[i] for i in index]
    # returning the predicted tag sequence
    return tag_seq


# ======================================================================================================================
# Function to calculate the maximum value in top k indices
# ======================================================================================================================

# getting list of indices and the matrix column as input
def get_max(new_index, column):
    # setting the first value as max
    max_ = column[new_index[0]]
    # iterating through list of indices
    for i in range(len(new_index)):
        # comparing value at each index to find new maximum
        if column[new_index[i]] > max_:
            # update the max_ if new maximum found
            max_ = column[new_index[i]]
    # returning the maximum value
    return max_


# ======================================================================================================================
# Function implementing viterbi search
# ======================================================================================================================

def viterbi(sentence, weights, cw_ct_count):
    # declaring vit matrix to store the score of each tuple
    vit = np.zeros((len(all_tags), len(sentence)))
    # declaring back pointer matrix to store the index of maximum value for each tuple
    back = np.zeros((len(all_tags), len(sentence)))

    # calculating the score for each word in the sentence with all possible tags
    for n in range(len(sentence)):
        for y in range(len(all_tags)):
            pair = [(sentence[n], all_tags[y])]
            # calculating phi for each tuple
            phi = phi_1(pair, cw_ct_count)
            # for first column
            if n == 0:
                # calculating and storing the score of the tuple
                vit[y][n] = phi * weights[(sentence[n], all_tags[y])]
            # for rest of the columns
            elif n >= 1:
                # finding the maximum value from the previous column
                max = np.amax(vit[:, n - 1])
                # calculating and storing the score of the tuple
                vit[y][n] = max + phi * weights[(sentence[n], all_tags[y])]
                # storing the index of maximum value in the back pointer matrix
                back[y][n] = np.argmax(vit[n - 1])
    # finding the list of indices of maximum values in each column
    index = np.argmax(vit, axis=0)
    # converting the list of indices to the respective tag sequence
    tag_seq = [all_tags[i] for i in index]
    # returning the predicted tag sequence
    return tag_seq


# ======================================================================================================================
# Function to plot the learning rate
# ======================================================================================================================

# learn list contains the no. of errors encountered in each iteration
def plot(learn):
    plt.plot(learn, marker='o', label='False Predictions')
    plt.xlabel('Iteration-Learning Progress')
    plt.ylabel('Number of Errors on training data')
    plt.legend()
    plt.grid()
    plt.show()


# ======================================================================================================================
# MAIN
# ======================================================================================================================
if __name__ == '__main__':

    # function calls for viterbi with argument '-v'
    if sys.argv[1] == '-v':
        print("\n--------------Running Viterbi Algorithm----------------) \n")

        # creating the feature space
        cw_ct_count = cw_ct_counts(train_data, freq_thresh=feat_red)

        print("\nTraining the perceptron with (cur_word, cur_tag) \n")
        # training the model with viterbi for train data
        weights, learn= train_perceptron(train_data, cw_ct_count, epochs=depochs, feature = 1)

        print("\nTesting and evaluating the perceptron with (cur_word, cur_tag) \n")
        # testing the model with viterbi for test data
        correct_tags, predicted_tags = test_perceptron(test_data, cw_ct_count,  weights, feature = 1)

        # evaluating our model with viterbi
        f1 = evaluate(correct_tags, predicted_tags)

        # plotting the learning rate
        plot(learn)

    # function calls for beam search with argument '-b'
    elif sys.argv[1] == '-b':
        print("\n--------------Running Beam Search Algorithm----------------) \n")

        print("Default beam value is 1-------------")
        # creating the feature space
        cp_ct_count = cw_ct_counts(train_data, freq_thresh=feat_red)

        print("\nTraining the perceptron with (cur_word, cur_tag) \n")
        # training the model with beam search for train data
        beam_weights, learn= train_perceptron(train_data, cp_ct_count, epochs=depochs, feature = 2)

        print("\nTesting and evaluating the perceptron with (cur_word, cur_tag) \n")
        # testing the model with viterbi for test data
        correct, predicted = test_perceptron(test_data, cp_ct_count,  beam_weights, feature = 2)

        # evaluating our model with viterbi
        f2 = evaluate(correct, predicted)

        # plotting the learning rate
        plot(learn)
