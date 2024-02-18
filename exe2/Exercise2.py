import nltk
from collections import defaultdict
import re


# DATASET_FILE = 'dataset.plk'


# Auxiliary functions
def fix_tag(tag):
    deliminators = r'[-+]'
    if tag != '--':
        tag = re.split(deliminators, tag)[0]
    while len(tag) > 1 and (tag[-1] == '$' or tag[-1] == '*'):
        tag = tag[:-1]

    return tag


def load_dataset(category):
    from nltk.corpus import brown

    tagged_sentences = brown.tagged_sents(categories=category)
    split_index = int(len(tagged_sentences) * 0.9)

    # Fix tags in training and test sets
    fixed_tagged_sentences = list()
    for i, sentence in enumerate(tagged_sentences):
        modified_sentence = [(word, fix_tag(tag)) for word, tag in sentence]
        fixed_tagged_sentences.append(modified_sentence)

    # Flatten the list of sentences
    train_set = fixed_tagged_sentences[:split_index]
    test_set = fixed_tagged_sentences[split_index:]

    return train_set, test_set


def get_all_pos_tags():
    all_pos_tags = set()
    for _, tag in nltk.corpus.brown.tagged_words():
        all_pos_tags.add(fix_tag(tag))

    return all_pos_tags


def get_vocabulary(tagged_sentences):
    vocabulary = set()
    for sentence in tagged_sentences:
        for word, _ in sentence:
            vocabulary.add(word)

    return vocabulary


def compute_most_likely_tags(train_set):
    # Create a dictionary of dictionaries: [word] [tag] [ number of times word was classified as this tag]
    word_tags = defaultdict(lambda: defaultdict(int))

    # Go over tuples received from brown of a word and it's true tag ('fact', ' NN')
    for sentence in train_set:

        for word, tag in sentence:
            word_tags[word][tag] += 1

    # Go over the three level dict and create normal dict { word: most counted tag}
    max_tag_dict = defaultdict(lambda: defaultdict(int))
    for word, tags in word_tags.items():
        max_tag_dict[word] = max(tags, key=tags.get)

    return max_tag_dict


def compute_error_rate(test_set, most_likely_tags):
    known_words_incorrect_tags = 0
    unknown_words_incorrect_tags = 0

    known_words = 0
    unknown_words = 0
    for sentence in test_set:
        for word, tag in sentence:
            # Appears in the training
            if word in most_likely_tags:
                known_words += 1
                likely_tag = most_likely_tags[word]
                if likely_tag != tag:
                    known_words_incorrect_tags += 1
            # Does not appear in the training -> if not NN is incorrect
            else:
                unknown_words += 1
                if tag != "NN":
                    unknown_words_incorrect_tags += 1

    print("MLE Error rate for known words: {}".format(known_words_incorrect_tags / known_words))
    print("MLE Error rate for un-known words: {}".format(unknown_words_incorrect_tags / unknown_words))
    print("MLE General error rate: {}\n".format((unknown_words_incorrect_tags + known_words_incorrect_tags) /
                                              (unknown_words + known_words)))


def compute_transition(train_set):
    # Create a dictionary of dictionaries: [tag1] [tag2] [ number of times tag2 appeared after tag1]
    # Tags start and stop are included
    transition_probs = defaultdict(lambda: defaultdict(int))
    start_tag = 'start'
    stop_tag = 'stop'

    for sentence in train_set:

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


def compute_emission(train_set, one_smoothing=False):
    # Create a dictionary of dictionaries: [tag] [word] [ number of times word was classified as this tag]
    emission_probs = defaultdict(lambda: defaultdict(int))
    train_words = get_vocabulary(train_set)

    for sentence in train_set:

        for word, tag in sentence:
            emission_probs[tag][word] += 1

    for tag in emission_probs.keys():
        total_num = 0
        for word in emission_probs[tag].keys():
            total_num += emission_probs[tag][word]

        for word in train_words:
            if one_smoothing:
                emission_probs[tag][word] = (emission_probs[tag][word] + 1)/(total_num + len(train_words))
            else:
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


def viterbi_algorithm(sentence, transition_probs, emission_probs, vocabulary, all_tags):
    """
    Run Viterbi algorithm to calculate predicted tags
    :param sentence: A list of words
    :param transition_probs: A dictionary contain all transition probabilities
    :param emission_probs: A dictionary contain all emission probabilities
    :param vocabulary: A list of all possible words witness on emission dictionary
    :param all_tags: A list of all possibles tags in brown corpus
    :return: A list of predicted tags
    """

    pi_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bp_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    n = len(sentence)
    start_tag = 'start'
    stop_tag = 'stop'
    noun_tag = 'NN'
    # In each iteration, we multiply by 10 to prevent the probability from becoming zero
    # due to the accumulation of numerous values smaller than one
    ten = 10

    # k = 1
    if sentence[0] in vocabulary:
        for v in all_tags:
            pi_dict[1][start_tag][v] = transition_probs[start_tag][v] * \
                                       emission_probs[v][sentence[0]] * ten
    else:
        for v in all_tags:
            if v == noun_tag:
                pi_dict[1][start_tag][v] = 1
            else:
                pi_dict[1][start_tag][v] = 0

    # k = 2
    if n >= 2 and sentence[1] in vocabulary:
        for u in all_tags:
            for v in all_tags:
                bp_dict[2][u][v] = start_tag
                pi_dict[2][u][v] = pi_dict[1][start_tag][u] * transition_probs[u][v] * \
                                   emission_probs[v][sentence[1]] * ten
    else:
        for u in all_tags:
            for v in all_tags:
                bp_dict[2][u][v] = start_tag
                if v == noun_tag:
                    pi_dict[2][u][v] = pi_dict[1][start_tag][u]
                else:
                    pi_dict[2][u][v] = 0

    # 2 < k < n + 1
    for k in range(3, n + 1):
        if sentence[k-1] in vocabulary:
            for u in all_tags:
                w = find_arg_max(pi_dict, k-1, u)
                for v in all_tags:
                    bp_dict[k][u][v] = w
                    pi_dict[k][u][v] = pi_dict[k-1][w][u] * transition_probs[u][v] * \
                                       emission_probs[v][sentence[k-1]] * ten
        else:
            for u in all_tags:
                w = find_arg_max(pi_dict, k-1, u)
                for v in all_tags:
                    bp_dict[k][u][v] = w
                    if v == noun_tag:
                        pi_dict[k][u][v] = pi_dict[k-1][w][u]
                    else:
                        pi_dict[k][u][v] = 0

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
        tags_result.insert(0, bp_dict[k+2][tags_result[0]][tags_result[1]])

    if n == 1:
        return tags_result[1:]

    return tags_result


def compute_error_rate_hmm(model_name, all_tags, train_words, test_set, transition_probs, emission_probs,
                           words_need_pseudo_replace={}):
    """
    Call Viterbi algorithm and prints error rates for known/unknown/all words from the train set
    :param model_name: Name of the model
    :param all_tags: A list of all possibles tags in brown corpus
    :param train_words: All original words belongs to the training set
    :param test_set: The training set
    :param transition_probs: A dictionary contain all transition probabilities
    :param emission_probs: A dictionary contain all emission probabilities
           (emission's vocabulary may be different from the training set)
    :param words_need_pseudo_replace: All words needed to replace with some pseudo word
    :return: VOID
    """
    known_words_incorrect_tags = 0
    unknown_words_incorrect_tags = 0
    known_words = 0
    unknown_words = 0

    emission_vocabulary = set()
    for tag in emission_probs.keys():
        for word in emission_probs[tag].keys():
            if emission_probs[tag][word] > 0:
                emission_vocabulary.add(word)

    for sentence in test_set:
        sentence_words = list()
        real_tags = list()

        for word, tag in sentence:
            if word in words_need_pseudo_replace:
                sentence_words.append(classify_word(word))
            else:
                sentence_words.append(word)

            real_tags.append(tag)

        predicted_tags = viterbi_algorithm(sentence_words, transition_probs, emission_probs, emission_vocabulary, all_tags)
        assert(len(real_tags) == len(predicted_tags))

        for k in range(len(predicted_tags)):
            word = sentence[k][0]
            if word in train_words:
                known_words += 1
                if predicted_tags[k] != real_tags[k]:
                    known_words_incorrect_tags += 1
            else:
                unknown_words += 1
                if predicted_tags[k] != real_tags[k]:
                    unknown_words_incorrect_tags += 1

        #index += 1

    if known_words > 0:
        print("{} error rate for known words: {}".format(model_name, known_words_incorrect_tags / known_words))
    if unknown_words > 0:
        print("{} error rate for un-known words: {}".format(model_name, unknown_words_incorrect_tags / unknown_words))
    if known_words + unknown_words > 0:
        print("{} general error rate: {}\n".format(model_name, (unknown_words_incorrect_tags + known_words_incorrect_tags) /
              (unknown_words + known_words)))


 # --------------------------------------------- * Noa - new methods  *  ---------------------------------------------
######################### NOA ########################


# do I need to take off , ! stuff like that
"""Go over all items in group count how many times they appeared """


def count_words_frequency(input_set):
    words_frequency = defaultdict(int)
    for sentence in input_set:
        for tup in sentence:
            word = tup[0]

            words_frequency[word] += 1
    return words_frequency


"""" Get a word and find it a pseudo tag classification """""


def classify_word(word):
    classes = []

    # Check for two-digit number
    if re.fullmatch(r'\d{2}', word):
        classes.append('twoDigitNum')

    # Check for four-digit number
    if re.fullmatch(r'\d{4}', word):
        classes.append('fourDigitNum')

    # Check for a word containing both digits and alphabets
    if re.search(r'\d', word) and re.search(r'[a-zA-Z]', word):
        classes.append('containsDigitandAlpha')

    # Check for all capital letters
    if word.isupper():
        classes.append('allCaps')

    # check for a prefix of big letter  [ places names etc ]
    else:
        if word[0].isupper():
            classes.append('startsWithCapital')

    # we might want to address the usecase where there is more than one class wuits for word ( we can use
    # our method here
    classes.append('NA')  # no entity
    return classes[0]


"""Get a set go over it, if you encounter low frequency word -> change tags 
Keep the structure of sentences and tuples """


def replace_pseudo_words(sentence, words_need_pseudo_replace):
    sentence_with_pseudo_words = []

    for word, tag in sentence:
        if word in words_need_pseudo_replace:
            pseudo_word = classify_word(word)
            sentence_with_pseudo_words.append((pseudo_word, tag))
        else:
            sentence_with_pseudo_words.append((word, tag))

    return sentence_with_pseudo_words


if __name__ == '__main__':
    # Download the Brown corpus if it's not already downloaded
    nltk.download('brown')

    # Question 3(a) + initiate important variables
    train_set, test_set = load_dataset(category='news')
    train_words = get_vocabulary(train_set)
    test_words = get_vocabulary(test_set)
    all_tags = get_all_pos_tags()

    # Print the number of sentences in the train and test sets
    # print("Number of sentences in train set:", len(train_set))
    # print("Number of sentences in test set:", len(test_set))

    # Question  3(b i)
    most_likely_tags = compute_most_likely_tags(train_set)

    # Question  3(b ii)
    compute_error_rate(test_set, most_likely_tags)

    # Question 3(c i )
    transition_probs = compute_transition(train_set)
    emission_probs = compute_emission(train_set)

    # Question 3(c iii)
    model_name = "HMM-Bigram"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, transition_probs, emission_probs)

    # Question 4(d i)
    new_emission_probs = compute_emission(train_set, one_smoothing=True)

    # Question 4(d ii)
    model_name = "HMM-Bigram-Laplace"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, transition_probs, new_emission_probs)

    # ---------------------------------------  QUESTION  E  i --------------------------------------------------------------

    words_frequency = count_words_frequency(train_set)
    frequency_thresh_hold = 3
    low_frequency_words = {k for k, v in words_frequency.items() if v < frequency_thresh_hold}
    words_need_pseudo_replace = low_frequency_words | (test_words - train_words)

    # --------------------------------------  QUESTION  E  ii   ------------------------------------------------------------

    # replace words with pseudo words
    pseudo_train_set = []
    for sentence in train_set:
        res = replace_pseudo_words(sentence, words_need_pseudo_replace)
        if res:
            pseudo_train_set.append(res)

    # Now we finished transitioning our corpus to pseudo indicators instead low frequency words we need
    # to recalculate the probabilities.

    pseudo_transition_probs = compute_transition(pseudo_train_set)
    pseudo_emission_probs = compute_emission(pseudo_train_set)

    # Compute error rates with pseudo words smoothing
    model_name = "HMM-Bigram-Pseudo"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, pseudo_transition_probs, pseudo_emission_probs,
                           words_need_pseudo_replace)

    # ---------------------------------  QUESTION  E  iii -----------------------------------------------------------

    # add add one smoothing

    # [i = predicted tag i][j = predicted tag j] [ number of tokens which have a true tag i and a predicted tag j]
    matrix = defaultdict(lambda: defaultdict(int))

