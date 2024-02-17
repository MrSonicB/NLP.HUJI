import spacy
import math
import nltk
from collections import defaultdict
import re


# DATASET_FILE = 'dataset.plk'


# Auxiliary functions
def load_dataset(category):
    from nltk.corpus import brown

    tagged_sentences = brown.tagged_sents(categories='news')
    split_index = int(len(tagged_sentences) * 0.9)

    # Flatten the list of sentences
    train_set = tagged_sentences[:split_index]
    test_set = tagged_sentences[split_index:]

    return train_set, test_set


# Helper function
# ---> don't think we need this
# def tag_sentence(sentence, most_likely_tags):
#     return [(word, most_likely_tags.get(word, 'NN')) for word in sentence]


def compute_most_likely_tags(training_set):
    # Create a dictionary of dictionaries: [word] [tag] [ number of times word was classified as this tag]
    word_tags = defaultdict(lambda: defaultdict(int))

    # Go over tuples received from brown of a word and it's true tag ('fact', ' NN')
    for sentence in training_set:

        for word, tag in sentence:
            word_tags[word][tag] += 1

    # Go over the three level dict and create normal dict { word: most counted tag}
    max_tag_dict = defaultdict(lambda: defaultdict(int))
    for word, tags in word_tags.items():
        max_tag_dict[word] = max(tags, key=tags.get)

    return max_tag_dict


def compute_error_rate(test_set, most_likely_tags):
    incorrect_known_tags = 0
    incorrect_unknown_tags = 0

    unknown_tags = 0
    known_tags = 0
    for sentence in test_set:
        for word, tag in sentence:
            # Appears in the training
            if word in most_likely_tags:
                known_tags += 1
                likely_tag = most_likely_tags[word]
                if likely_tag != tag:
                    incorrect_known_tags += 1
            # Does not appear in the training -> if not NN is incorrect
            else:
                unknown_tags += 1
                if tag != "NN":
                    incorrect_unknown_tags += 1

    print("Error rate for un-known words", incorrect_unknown_tags / unknown_tags)
    print("Error rate for known words", incorrect_known_tags / known_tags)
    print("General error rate", (incorrect_unknown_tags + incorrect_known_tags) / (unknown_tags + known_tags))


def compute_tags_unigram(training_set):
    # create dictionary [tag] [ count in training set)
    tag_counter = 0
    tag_unigram_counts = defaultdict(lambda : 0 )
    for sentence in training_set:
        for word, tag in sentence:
            if tag_unigram_counts[tag] == 0:
                tag_counter += 1  # first time tag added to dict
            tag_unigram_counts[tag] += 1

    tag_unigram_prob = defaultdict(lambda :float)
    for tag, count in tag_unigram_counts.items():

        tag_unigram_prob[tag] = count / tag_counter #-> why?

    # print(tag_unigram_prob.items())
    return tag_unigram_prob


def compute_emission2(training_set):
    probs_unigram = compute_tags_unigram(training_set)
    # Create a dictionary of dictionaries: [previous tag] [tag] [ number of times tag2 appeared after tag1 ]
    tag_bigram_counts = defaultdict(lambda: defaultdict(int))
    for sentence in training_set:
        previous_tag = None
        for word, tag in sentence:

            if previous_tag is not None:
                tag_bigram_counts[previous_tag][tag] += 1

            previous_tag = tag

    # tag_bigram_probs = defaultdict(lambda: defaultdict(float))
    # for previous_tag, tag in tag_bigram_counts:
    #     prob_prev = probs_unigram[previous_tag]
    #     prob_tranision
    #     tag_bigram_probs[previous_tag][tag] = prob_

    #need aonther counter or something

    # print(tag_bigram_counts)

############################ Bar ###################################
def get_all_pos_tags():
    all_tags = set()
    deliminators = r'[-+]'
    for _, tag in nltk.corpus.brown.tagged_words():
        if tag != '--':
            tag = re.split(deliminators, tag)[0]
        while len(tag) > 1 and (tag[-1] == '$' or tag[-1] == '*'):
            tag = tag[:-1]

        all_tags.add(tag)

    all_tags = list(all_tags)
    return all_tags


def compute_transition(training_set):
    # Create a dictionary of dictionaries: [tag1] [tag2] [ number of times tag2 appeared after tag1]
    # Tags start and stop are included
    transition_probs = defaultdict(lambda: defaultdict(int))
    start_tag = '*'
    stop_tag = 'stop'

    for sentence in training_set:

        prev_tag = start_tag
        for _, tag in sentence:
            transition_probs[prev_tag][tag] += 1
            prev_tag = tag

        transition_probs[prev_tag][stop_tag] += 1

    for prev_tag in transition_probs.keys():
        total_num = 0
        for tag in transition_probs[prev_tag].keys():
            total_num += transition_probs[prev_tag][tag]

        for tag in transition_probs[prev_tag].keys():
            transition_probs[prev_tag][tag] /= total_num

    return transition_probs


def compute_emission(training_set):
    # Create a dictionary of dictionaries: [tag] [word] [ number of times word was classified as this tag]
    emission_probs = defaultdict(lambda: defaultdict(int))

    for sentence in training_set:

        for word, tag in sentence:
            emission_probs[tag][word] += 1

    for tag in emission_probs.keys():
        total_num = 0
        for word in emission_probs[tag].keys():
            total_num += emission_probs[tag][word]

        for word in emission_probs[tag].keys():
            emission_probs[tag][word] /= total_num

    return emission_probs


def find_arg_max(pi_dict, k, u):
    # Find the string w maximizing dictionary[k, w, u]
    max_w = None
    max_value = float('-inf')  # Initialize to negative infinity

    for w in pi_dict[k].keys():
        # Check if a[k, w, u] is greater than the current maximum value
        if pi_dict[k][w][u] > max_value:
            max_value = pi_dict[k][w][u]
            max_w = w

    return max_w


def viterbi_algorithm(sentence, transition_probs, emission_probs, seen_words, all_tags):
    #print("algorithm: {}".format(1))
    # dict with [k, w, u]
    pi_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bp_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    # Get the list of all possible tags
    n = len(sentence)
    start_tag = '*'
    stop_tag = 'stop'
    noun_tag = 'NN'

    #print("algorithm: {}".format(2))

    # k = 1
    if sentence[0] in seen_words:
        for v in all_tags:
            pi_dict[1][start_tag][v] = transition_probs[start_tag][v] * emission_probs[v][sentence[0]]
    else:
        for v in all_tags:
            if v == noun_tag:
                pi_dict[1][start_tag][v] = 1
            else:
                pi_dict[1][start_tag][v] = 0

    #print("algorithm: finished k=1")
    # k = 2
    if n >= 2 and sentence[1] in seen_words:
        for u in all_tags:
            for v in all_tags:
                bp_dict[2][u][v] = start_tag
                pi_dict[2][u][v] = pi_dict[1][start_tag][u] * transition_probs[u][v] * emission_probs[v][sentence[1]]
    else:
        for u in all_tags:
            for v in all_tags:
                bp_dict[2][u][v] = start_tag
                if v == noun_tag:
                    pi_dict[2][u][v] = pi_dict[1][start_tag][u]
                else:
                    pi_dict[2][u][v] = 0
    #print("algorithm: finished k=2")

    # 2 < k < n + 1
    for k in range(3, n + 1):
        if sentence[k-1] in seen_words:
            for u in all_tags:
                for v in all_tags:
                    w = find_arg_max(pi_dict, k-1, u)
                    bp_dict[k][u][v] = w
                    pi_dict[k][u][v] = pi_dict[k-1][w][u] * transition_probs[u][v] * emission_probs[v][sentence[k-1]]
        else:
            for u in all_tags:
                for v in all_tags:
                    w = find_arg_max(pi_dict, k-1, u)
                    bp_dict[k][u][v] = w
                    if v == noun_tag:
                        pi_dict[k][u][v] = pi_dict[k - 1][w][u]
                    else:
                        pi_dict[k][u][v] = 0
        #print("algorithm: finished k={}".format(k))

    prev_last_tag = None
    last_tag = None
    max_prob = float('-inf')  # Initialize to negative infinity

    for u in pi_dict[n]:
        for v in pi_dict[n][u]:
            prob = pi_dict[n][u][v] * transition_probs[v][stop_tag]
            if prob > max_prob:
                max_prob = prob
                prev_last_tag = u
                last_tag = v

    tags_result = list([prev_last_tag, last_tag])
    for k in range(n-2, 0, -1):
        tags_result.insert(0, bp_dict[k][tags_result[0]][tags_result[1]])

    return tags_result


def compute_error_rate_hmm(test_set, transition_probs, emission_probs, seen_words, all_tags):
    total_tags = 0
    incorrect_tags = 0

    index = 1
    for sentence in test_set:
        print("Sentence number {}".format(index))
        words = list()
        real_tags = list()
        for word, tag in sentence:
            words.append(word)
            real_tags.append(tag)

        predicted_tags = viterbi_algorithm(words, transition_probs, emission_probs, seen_words, all_tags)
        if len(real_tags) != len(predicted_tags):
            print("Sentence is: {}".format(words))
            print("Real tags: {}, length = {}".format(real_tags, len(real_tags)))
            print("Predicted tags: {}, length = {}".format(predicted_tags, len(predicted_tags)))
            return

        for k in range(len(predicted_tags)):
            if predicted_tags[k] != real_tags[k]:
                incorrect_tags += 1

        total_tags += len(real_tags)
        index += 1

    print("General error rate", incorrect_tags / total_tags)


if __name__ == '__main__':
    # Download the Brown corpus if it's not already downloaded
    nltk.download('brown')

    # Question 3(a)
    train_set, test_set = load_dataset(category='news')

    # Print the number of sentences in the train and test sets
    print("Number of sentences in train set:", len(train_set))
    print("Number of sentences in test set:", len(test_set))

    # Question  3(b i)

    #most_likely_tags = compute_most_likely_tags(train_set)

    # Question  3(b ii)
    #compute_error_rate(test_set, most_likely_tags)

    # Question 3(c i )
    transition_probs = compute_transition(train_set)
    emission_probs = compute_emission(train_set)

    # Question 3(c iii)
    seen_words = set()
    for tag in emission_probs.keys():
        seen_words.update(emission_probs[tag].keys())
    print("Number of seen words: {}".format(len(seen_words)))

    all_tags = get_all_pos_tags()
    print("Number possible POS tags: {}".format(len(all_tags)))
    compute_error_rate_hmm(test_set, transition_probs, emission_probs, seen_words, all_tags)





