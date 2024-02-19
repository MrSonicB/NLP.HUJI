import nltk
from collections import defaultdict
import re


# Auxiliary functions
def load_dataset(category):
    """
    1. Load the tagged brown corpus sentences with the desired category
    2. Fix tags it contain for each word
    3. Split to train and test corpus with ratio of 9:1
    :param category: The desired category of the brown corpus
    :return: The training and test corpus - A lists of sentences (List) containing tagged words (tuples of word and its tag)
    """
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


def fix_tag(tag):
    """
    Fix a tag according to the forum guidance e.g. AT$ -> AT, CD* -> CD, WEB+VR -> WEB, etc.
    :param tag: The tag we want to fix
    :return: The fixed tad
    """
    deliminators = r'[-+]'
    if tag != '--':
        tag = re.split(deliminators, tag)[0]
    while len(tag) > 1 and (tag[-1] == '$' or tag[-1] == '*'):
        tag = tag[:-1]

    return tag


def get_all_pos_tags():
    """
    Get all possible (fixed) tags of brown corpus
    :return: A set with all possibles tags in brown corpus
    """
    all_pos_tags = set()
    for _, tag in nltk.corpus.brown.tagged_words():
        all_pos_tags.add(fix_tag(tag))

    return all_pos_tags


def get_vocabulary(input_set):
    """
    Get the vocabulary of a given corpus
    :param input_set: The input corpus
    :return: A set with all different words seen in the input corpus
    """
    vocabulary = set()
    for sentence in input_set:
        for word, _ in sentence:
            vocabulary.add(word)

    return vocabulary


# Questions Functions
def compute_most_likely_tags(train_set):
    """
    For each word, find its most likely tag based on MLE
    :param train_set: The train corpus
    :return: A dictionary containing (word, most likely tag)
    """
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
    """
    Print the error rates for regular MLE
    :param test_set: The test corpus
    :param most_likely_tags: A dictionary containing (word, most likely tag)
    :return: Void
    """
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


def compute_transition(input_set):
    """
    Compute transition probabilities on the input corpus with respect to MLE
    :param input_set: The input corpus
    :return: A dictionary contain all transition probabilities -> transition_probs[tag1][tag2] = Pr(tag2 | tag1)
    """
    # Create a dictionary of dictionaries: [tag1] [tag2] [ number of times tag2 appeared after tag1]
    # Tags start and stop are included
    transition_probs = defaultdict(lambda: defaultdict(int))
    start_tag = 'start'
    stop_tag = 'stop'

    for sentence in input_set:

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


def compute_emission(input_set, one_smoothing=False):
    """
    Compute emission probabilities on the input corpus with respect to MLE
    :param input_set: The input corpus
    :param one_smoothing: A boolean enabled one-smoothing, default is false
    :return: A dictionary contain all emission probabilities -> emission_probs[tag][word] = Pr(word | tag)
    """
    # Create a dictionary of dictionaries: [tag] [word] [ number of times the word was classified with this tag]
    emission_probs = defaultdict(lambda: defaultdict(int))
    train_words = get_vocabulary(input_set)

    for sentence in input_set:

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
    """
    An auxiliary function for viterbi_algorithm: Find thd tag w such that pi(k,w,u) is maximal
    :param pi_dict: The Pi dictionary composed of [level][tag][next tag]
    :param k: The level to consider
    :param u: The next tag
    :return: argmax of pi(k,w,u)
    """
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
    # Initiate pi and back pointer dictionaries (as we saw at class)
    pi_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bp_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    n = len(sentence)
    start_tag = 'start'
    stop_tag = 'stop'
    noun_tag = 'NN'
    # In each iteration, we multiply by 10 to prevent the probability from becoming zero
    # due to the accumulation of numerous values smaller than one
    ten = 10

    # Split K for three cases: In each case we check if the word is seen in the vocabulary, otherwise we force the
    # algorithm to choose the tag 'NN' for unknown word
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
    if n >= 2:
        if sentence[1] in vocabulary:
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

    # Calculate most likely last two tags
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

    # Use back pointers to get the list of predicted tags
    tags_result = list([prev_last_tag, last_tag])
    for k in range(n-2, 0, -1):
        tags_result.insert(0, bp_dict[k+2][tags_result[0]][tags_result[1]])

    if n == 1:
        return tags_result[1:]

    return tags_result


def compute_error_rate_hmm(model_name, all_tags, train_words, test_set, transition_probs, emission_probs,
                           words_need_pseudo_replace=set()):
    """
    Call Viterbi algorithm and prints error rates for known/unknown/all words from the train set
    :param model_name: Name of the model
    :param all_tags: A list of all possibles tags in brown corpus
    :param train_words: All original words belongs to the training corpus
    :param test_set: The test corpus
    :param transition_probs: A dictionary contain all transition probabilities
    :param emission_probs: A dictionary contain all emission probabilities
           (emission's vocabulary may be different from the training set)
    :param words_need_pseudo_replace: All words needed to replace with some pseudo word, default is empy set
    :return: VOID
    """
    known_words_incorrect_tags = 0
    unknown_words_incorrect_tags = 0
    known_words = 0
    unknown_words = 0

    # Build a vocabulary for word seen in the emission probabilities: may be original training set or the
    # training set with all low frequencies words replaced with their pseudo word
    emission_vocabulary = set()
    for tag in emission_probs.keys():
        for word in emission_probs[tag].keys():
            if emission_probs[tag][word] > 0:
                emission_vocabulary.add(word)

    # For each sentence from the test corpus, build a words list with pseudo words and a tag list with all real tags
    for sentence in test_set:
        sentence_words = list()
        real_tags = list()

        for word, tag in sentence:
            if word in words_need_pseudo_replace:
                sentence_words.append(classify_word(word))
            else:
                sentence_words.append(word)

            real_tags.append(tag)

        # Call Viterbi algorithm to get a list with the predicted tags (verify it has the same length as real tags list)
        predicted_tags = viterbi_algorithm(sentence_words, transition_probs, emission_probs, emission_vocabulary, all_tags)
        assert(len(real_tags) == len(predicted_tags))

        # For each word in the original sentence (no pseudo), check if it is a known word and
        # compare the predicted tag to the real one
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

    # Print the calculated error rates
    if known_words > 0:
        print("{} error rate for known words: {}".format(model_name, known_words_incorrect_tags / known_words))
    if unknown_words > 0:
        print("{} error rate for un-known words: {}".format(model_name, unknown_words_incorrect_tags / unknown_words))
    if known_words + unknown_words > 0:
        print("{} general error rate: {}\n".format(model_name, (unknown_words_incorrect_tags + known_words_incorrect_tags) /
              (unknown_words + known_words)))


def count_words_frequency(input_set):
    """
    Compute word distribution over the given corpus
    :param input_set: The input corpus
    :return: A dictionary describing the words distribution
    """
    words_frequency = defaultdict(int)
    for sentence in input_set:
        for word, _ in sentence:
            words_frequency[word] += 1

    return words_frequency


def classify_word(word):
    """
    Compute the pseudo word of the given word
    :param word: The input word we wish to compute its pseudo word
    :return: The pseudo of the input word (The first match). The default pseudo is 'NA'
    """
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


def replace_pseudo_words(sentence, words_need_pseudo_replace):
    """
    Replace low frequency or unknown words with their pseudo word
    :param sentence: The sentence we want to rebuild (list of [(word, tag)])
    :param words_need_pseudo_replace: A list of all words (low frequency or unknown) we neet to replace with pseudo word
    :return: A new sentence where each low frequency or unknown word has replaced with its pseudo word
    """
    sentence_with_pseudo_words = []

    for word, tag in sentence:
        if word in words_need_pseudo_replace:
            pseudo_word = classify_word(word)
            sentence_with_pseudo_words.append((pseudo_word, tag))
        else:
            sentence_with_pseudo_words.append((word, tag))

    return sentence_with_pseudo_words


def create_confusion_matrix(test_set, transition_probs, emission_probs, vocabulary, all_tags, print_matrix=False):
    """
    Calculate confusion matrix and print 10 most frequents errors for predicting the wrong tag
    :param test_set: The test corpus set
    :param transition_probs: A dictionary contain all transition probabilities
    :param emission_probs: A dictionary contain all emission probabilities
    :param vocabulary: A list of all possible words witness on emission dictionary
    :param all_tags: A list of all possibles tags in brown corpus
    :param print_matrix: A verbose boolean used to print the confusion matrix, default is false
    :return: Void
    """
    # Initialize the confusion matrix -> [i = predicted tag i][j = predicted tag j] [ number of tokens which have a
    # true tag i and a predicted tag j]
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    # Iterate through the test set and predict tags using the Viterbi algorithm
    for sentence in test_set:
        words = [word for word, _ in sentence]
        real_tags = [tag for _, tag in sentence]
        predicted_tags = viterbi_algorithm(words, transition_probs, emission_probs, vocabulary, all_tags)

        # Update the confusion matrix
        for i in range(len(real_tags)):
            real_tag = real_tags[i]
            predicted_tag = predicted_tags[i]
            confusion_matrix[real_tag][predicted_tag] += 1

    if print_matrix:
        # Print the confusion matrix
        print("Confusion matrix:")
        print("\t" + "\t".join(all_tags))
        for real_tag in all_tags:
            row = [str(confusion_matrix[real_tag][predicted_tag]) for predicted_tag in all_tags]
            print(f"{real_tag}\t" + "\t".join(row))

    # Analyze the most frequent errors counts is going to be a list like this : [ ((real - i , predicted -j) ,
    #  m[i,j]) =  amount of times a word/token has gotten this tuple (i,j) -> where ((i,j), m[i,j]) is added
    # only if it is i!=j
    print("10 most frequent errors:")
    error_counts = []
    for real_tag in all_tags:
        for predicted_tag in all_tags:
            if real_tag != predicted_tag:
                error_counts.append(((real_tag, predicted_tag), confusion_matrix[real_tag][predicted_tag]))

    error_counts.sort(key=lambda x: x[1], reverse=True)
    for index, ((real_tag, predicted_tag), count) in enumerate(error_counts[:10]):
        print(f"{index + 1}. True tag: {real_tag}, Predicted tag: {predicted_tag}, Count: {count}")


# Main function
if __name__ == '__main__':
    # Download the Brown corpus if it's not already downloaded
    nltk.download('brown')

    # Question 3(a)
    # Load brown corpus, split to train and test sets and initiate other important variables
    train_set, test_set = load_dataset(category='news')
    train_words = get_vocabulary(train_set)
    test_words = get_vocabulary(test_set)
    all_tags = get_all_pos_tags()

    # Question  3(b i)
    # Compute for each word its most likely tag based on MLE
    most_likely_tags = compute_most_likely_tags(train_set)

    # Question  3(b ii)
    # Compute error rates for regular MLE
    compute_error_rate(test_set, most_likely_tags)

    # Question 3(c i )
    # Compute transition and emission probabilities
    transition_probs = compute_transition(train_set)
    emission_probs = compute_emission(train_set)

    # Question 3(c iii)
    # Compute error rates for HMM-Bigram
    model_name = "HMM-Bigram"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, transition_probs, emission_probs)

    # Question 3(d i)
    # Recalculate the emission probabilities when using one-smoothing
    smoothing_emission_probs = compute_emission(train_set, one_smoothing=True)

    # Question 3(d ii)
    # Compute error rates for HMM-Bigram with one-smoothing
    model_name = "HMM-Bigram-Laplace"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, transition_probs, smoothing_emission_probs)

    # Question 3(e i)
    words_frequency = count_words_frequency(train_set)
    frequency_thresh_hold = 3
    low_frequency_words = {k for k, v in words_frequency.items() if v < frequency_thresh_hold}
    words_need_pseudo_replace = low_frequency_words | (test_words - train_words)

    # Question 3(e ii)
    # Replace words with pseudo words
    pseudo_train_set = []
    for sentence in train_set:
        res = replace_pseudo_words(sentence, words_need_pseudo_replace)
        if res:
            pseudo_train_set.append(res)

    # After we finished transitioning our corpus to pseudo indicators instead low frequency words we need
    # to recalculate the transition and emission probabilities.
    pseudo_transition_probs = compute_transition(pseudo_train_set)
    pseudo_emission_probs = compute_emission(pseudo_train_set)

    # Compute error rates for HMM-Bigram with pseudo words
    model_name = "HMM-Bigram-Pseudo"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, pseudo_transition_probs, pseudo_emission_probs,
                           words_need_pseudo_replace)

    # Question 3(e iii)
    # Recalculate the emission probabilities when using pseudo and one-smoothing
    smoothing_pseudo_emission_probs = compute_emission(pseudo_train_set, one_smoothing=True)

    # Compute error rates for HMM-Bigram with pseudo words and one-smoothing
    model_name = "HMM-Bigram-Pseudo-Laplace"
    compute_error_rate_hmm(model_name, all_tags, train_words, test_set, pseudo_transition_probs,
                           smoothing_pseudo_emission_probs, words_need_pseudo_replace)

    # Build a confusion matrix and print 10 most frequents errors for predicting the wrong tag
    pseudo_vocabulary = set()
    for word in train_words:
        if word in words_need_pseudo_replace:
            pseudo_vocabulary.add(classify_word(word))
        else:
            pseudo_vocabulary.add(word)

    create_confusion_matrix(test_set, pseudo_transition_probs, smoothing_pseudo_emission_probs, pseudo_vocabulary,
                            all_tags)
