import spacy
from datasets import load_dataset
import pickle
import os
import math

DATASET_FILE = 'dataset.plk'


class UnigramLM:
    def __init__(self):
        self.probs = {}

    # Train model according to MLE
    def train(self, docs):
        for doc in docs:
            for token in doc:
                if token.is_alpha:
                    lemma = token.lemma_
                    if lemma not in self.probs.keys():
                        self.probs[lemma] = 1
                    else:
                        self.probs[lemma] += 1

        # calculate the actual probabilities
        total_sum = 0
        for lemma in self.probs.keys():
            total_sum += self.probs[lemma]

        for lemma in self.probs.keys():
            self.probs[lemma] /= total_sum

    # Calculate probability in log space
    def evaluate(self, doc):
        log_prob = 0

        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_
                if lemma in self.probs.keys():
                    log_prob += math.log(self.probs[lemma])
                else:
                    return float('-inf')

        return log_prob


class BigramLM:
    def __init__(self):
        self.start_lemma = nlp('^')[0].lemma_
        self.probs = {}

    # Train model according to MLE
    def train(self, docs):
        # For memory reasons I choose to use dictionary rather than large sparse matrix
        self.probs[self.start_lemma] = {}

        for doc in docs:
            prev_lemma = self.start_lemma
            for token in doc:
                if token.is_alpha:

                    if prev_lemma not in self.probs.keys():
                        self.probs[prev_lemma] = {}

                    lemma = token.lemma_
                    if lemma not in self.probs[prev_lemma].keys():
                        self.probs[prev_lemma][lemma] = 1
                    else:
                        self.probs[prev_lemma][lemma] += 1

                    prev_lemma = lemma

        # calculate the actual probabilities
        for lemma in self.probs.keys():
            total_sum = 0
            for next_lemma in self.probs[lemma].keys():
                total_sum += self.probs[lemma][next_lemma]

            for next_lemma in self.probs[lemma].keys():
                self.probs[lemma][next_lemma] /= total_sum

    # Calculate probability in log space
    def evaluate(self, doc):
        log_prob = 0
        prev_lemma = self.start_lemma

        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_
                if prev_lemma in self.probs.keys() and lemma in self.probs[prev_lemma].keys():
                    log_prob += math.log(self.probs[prev_lemma][lemma])
                    prev_lemma = lemma
                else:
                    return float('-inf')

        return log_prob


class LIS_LM():
    def __init__(self, lambda_U, lambda_B):
        self.lambda_U = lambda_U
        self.lambda_B = lambda_B
        self.unigram_model = UnigramLM()
        self.bigram_model = BigramLM()

    def train(self, docs):
        self.unigram_model.train(docs)
        self.bigram_model.train(docs)

    def evaluate(self, doc):
        log_prob = 0
        prev_lemma = self.bigram_model.start_lemma

        for token in doc:
            if token.is_alpha:
                lemma = token.lemma_
                LI_prob = 0
                if lemma in self.unigram_model.probs.keys():
                    LI_prob += self.lambda_U * self.unigram_model.probs[lemma]
                if prev_lemma in self.bigram_model.probs.keys() and lemma in self.bigram_model.probs[prev_lemma].keys():
                    LI_prob += self.lambda_B * self.bigram_model.probs[prev_lemma][lemma]

                if LI_prob == 0:
                    return float('-inf')
                else:
                    log_prob += math.log(LI_prob)

        return log_prob


# Auxiliary functions
def download_and_save_dataset_to_local_file():
    from tqdm import tqdm
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    texts = dataset["text"]
    docs = [nlp(texts[i]) for i in tqdm(range(len(texts)))]

    with open(DATASET_FILE, 'wb+') as f:
        pickle.dump(docs, f)


def load_dataset_from_local_file():
    with open(DATASET_FILE, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # Load dataset
    nlp = spacy.load("en_core_web_sm")
    if not os.path.exists(DATASET_FILE):
        download_and_save_dataset_to_local_file()

    docs = load_dataset_from_local_file()
    print("Finished loading dataset")

    # Initiate Languages modules
    unigram = UnigramLM()
    bigram = BigramLM()

    # Question 1
    print("Question 1:")

    # Train Language modules
    unigram.train(docs)
    print("Finished training unigram model")
    bigram.train(docs)
    print("Finished training bigram model")
    print("")

    # Question 2
    print("Question 2:")

    max = 0
    lemma = nlp("in")[0].lemma_
    if lemma in bigram.probs.keys():
        max_prob = -1
        chosen_lemma = ""
        for next_lemma in bigram.probs[lemma]:
            prob = bigram.probs[lemma][next_lemma]
            if prob > max_prob:
                max_prob = prob
                chosen_lemma = next_lemma

        print("I have a house in {}".format(chosen_lemma))
        print("The answer is the word: '{}'".format(chosen_lemma))
    else:
        print("All lemmas has zero probability")

    print("")

    # Question 3
    print("Question 3:")

    dos1 = nlp("Brad Pitt was born in Oklahoma")
    dos2 = nlp("The actor was born in USA")

    # 3a: Computing the probability of each sentence separately using Bigram model
    print("Probability for sentence 1: {}".format(math.exp(bigram.evaluate(dos1))))
    print("Probability for sentence 2: {}".format(math.exp(bigram.evaluate(dos2))))

    # 3b: Computing the perplexity of both the following two sentences using Bigram model
    num_of_tokens = 0
    for dos in [dos1, dos2]:
        for token in dos:
            if token.is_alpha:
                num_of_tokens += 1

    l = (bigram.evaluate(dos1) + bigram.evaluate(dos2))/num_of_tokens
    print("Perplexity for both sentences is: {}".format(math.exp(-1 * l)))
    print("")

    # Question 4
    print("Question 4:")

    lis_model = LIS_LM(lambda_U=1/3, lambda_B=2/3)
    lis_model.train(docs)
    print("Finished training linear interpolation smoothing model")

    # 4a: Computing the probability of each sentence separately using Linear interpolation smoothing model
    print("Probability for sentence 1: {}".format(math.exp(lis_model.evaluate(dos1))))
    print("Probability for sentence 2: {}".format(math.exp(lis_model.evaluate(dos2))))

    # 4b: Computing the perplexity of both the following two sentences using Linear interpolation smoothing model
    l = (lis_model.evaluate(dos1) + lis_model.evaluate(dos2))/num_of_tokens
    print("Perplexity for both sentences is: {}".format(math.exp(-1 * l)))
