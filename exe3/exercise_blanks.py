import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300
HIDDEN_STATES_DIM = 100
BATCH_SIZE = 64

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
RARE_WORDS = "rare_words"
NEGATED_POLARITY = "negated_polarity"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)

    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    *Note*
    Handling embedding_dim that is not 300 (i.e., W2V_EMBEDDING_DIM):
      If W2V_EMBEDDING_DIM > embedding_dim -> Truncate the embedding vector to size of embedding_dim
      If W2V_EMBEDDING_DIM < embedding_dim -> Pad the embedding vector with zeros in the last indices
    """
    min_embedding_dim = min(W2V_EMBEDDING_DIM, embedding_dim)
    result = np.zeros(embedding_dim, dtype=np.float32)

    for word in sent.text:
        if word in word_to_vec.keys():
            word_embedding = np.zeros(embedding_dim, dtype=np.float32)
            word_embedding[:min_embedding_dim] = word_to_vec[word][:min_embedding_dim]
            result += word_embedding

    result = result / len(sent.text)
    return result


# After we have the index for each word in the vocabulary (get_word_to_ind(words_list) )
# we can move forward to create the one-hot embedding for each word, and to average all the embeddings of the words
# in the sentence.
# Complete the following methods in order to assist you in calculating the average
# one-hot embedding:
#  𝒈𝒆𝒕_𝒐𝒏𝒆_𝒉𝒐𝒕(𝒔𝒊𝒛𝒆,𝒊𝒏𝒅),  𝒂𝒗𝒆𝒓𝒂𝒈𝒆_𝒐𝒏𝒆_𝒉𝒐𝒕𝒔(𝒔𝒆𝒏𝒕,𝒘𝒐𝒓𝒅_𝒕𝒐_𝒊𝒏𝒅)

def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    array = np.zeros(size, dtype=np.float32)
    array[ind] = 1
    return array


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return: returns the average one-hot embedding of the tokens in the sentence
    """
    vocab_size = len(word_to_ind.keys())
    sum_array = np.zeros(vocab_size, dtype=np.float32)
    for word in sent.text:
        index = word_to_ind[word]
        sum_array[index] += 1

    avg_array = sum_array / len(sent.text)
    return avg_array


# "Therefore, to create a one-hot embedding, we must first assign an index to each word in
# the vocabulary.
# 2. Complete the 𝑔𝑒𝑡_𝑤𝑜𝑟𝑑_𝑡𝑜_𝑖𝑛𝑑(𝑤𝑜𝑟𝑑_𝑐𝑜𝑢𝑛𝑡) method. This method will be used in
# order to create a mapping between words in the vocabulary, and their assigned
# indices."
# Creating the vectors, going over all the vocabulary and giving indexes:
def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """

    indexed_words_dict = {word: index for index, word in enumerate(words_list)}
    return indexed_words_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: a numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    *Note*
    Handling embedding_dim that is not 300 (i.e., W2V_EMBEDDING_DIM):
      If W2V_EMBEDDING_DIM > embedding_dim -> Truncate the embedding vector to size of embedding_dim
      If W2V_EMBEDDING_DIM < embedding_dim -> Pad the embedding vector with zeros in the last indices
    """
    min_embedding_dim = min(W2V_EMBEDDING_DIM, embedding_dim)
    result = np.zeros((seq_len, embedding_dim), dtype=np.float32)

    for i, word in enumerate(sent.text):
        if i == seq_len:
            break  # Truncate the sentence
        if word in word_to_vec.keys():
            result[i, :min_embedding_dim] = word_to_vec[word][:min_embedding_dim]

    return result


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50, embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """
        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)

        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # Start of new added code
        self.sentences[RARE_WORDS] = list(self.sentences[TEST][i] for i in data_loader.get_rare_words_examples(
            sentences_list=self.sentences[TEST], dataset=self.sentiment_dataset))
        self.sentences[NEGATED_POLARITY] = list(self.sentences[TEST][i] for i in data_loader.get_negated_polarity_examples(
            sentences_list=self.sentences[TEST]))
        # End of new added code

        # map data splits to sentence input preparation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))

        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN, VAL, TEST, RARE_WORDS and NEGATED_POLARITY
        :return: torch batches iterator for this part of the dataset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN, VAL, TEST, RARE_WORDS and NEGATED_POLARITY
        :return: numpy array with the labels of the requested part of the dataset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------


class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        # Initialize the LSTM model
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)

        # Initialize a Dropout layer to randomly drop some of the input's features
        self.dropout = nn.Dropout(dropout)

        # Initialize a Linear layer (fully connected) with input size of 2 hidden_dim as
        # we concatenate the two hidden states from both directions
        self.linear = nn.Linear(hidden_dim * 2, 1, bias=True)

    def forward(self, text):
        """
        Initiate the forward stage of our LSTM model
        :param text: A tensor of shape (batch_size, SEQ_LEN, embedding_dim) that resemble the embedding vectors for
                     each word of each sentence in the batch.
        :return: A tensor of shape (batch_size, 1) that resemble the logit values to receive a positive sentiment for
                 each sentence in the batch.
        """
        """
        
        lstm_out, _ = self.lstm(text)

        # From the last LSTM layer, get the concatenation of the last hidden states from both directions
        hidden_concat_last_layer = lstm_out[:, -1, :]
        """
        # Pytorch's built-in LSTM forward stage
        _, (hidden, _) = self.lstm(text)

        # Concatenate the actually final hidden states from both directions
        hidden_forward = hidden[0, :, :]  # Last forward direction hidden state
        hidden_backward = hidden[1, :, :]  # Last backward direction hidden state
        hidden_concat_last_layer = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Forward the concentrated vector through the Dropout layer
        dropped_hidden_concat_last_layer = self.dropout(hidden_concat_last_layer)

        # Forward the concentrated vector through the linear layer
        logits = self.linear(dropped_hidden_concat_last_layer)

        # Since we use binary cross entrpy loss that accept logits, we just return the logit from the linear layer
        # without passing through Sigmoid activation function
        return logits

    def predict(self, text):
        """
        Predict the sentiment for each sentence in the batch:
        0 - Negative sentiment
        1 - Positive sentiment
        :param text: A tensor of shape (batch_size, SEQ_LEN, embedding_dim), resemble the embedding vectors for each
                     word of each sentence in the batch.
        :return: A tensor of shape (batch_size, 1), resemble the sentiment (0 or 1) of each sentence in the batch.
        """
        # Use forward function to get logic value for each sentence
        logits_tensor = self(text)

        # Use sigmoid to convert logics to probabilities (for a positive sentiment)
        probs_tensor = torch.sigmoid(logits_tensor)

        # Round probabilities to 0 or 1 to get sentiment decisions to each sentence in the batch
        prediction = torch.round(probs_tensor)
        return prediction


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    # X supposed to be a vector representing the input data
    def forward(self, x):
        x = x.float()  # cast to float32 to avoid different floats
        return self.linear(x)

    def predict(self, x):
        fr_x = self(x)
        prob = torch.sigmoid(fr_x)
        prediction = prob.round()  # Will round to the closer value 0/1
        return prediction


# ------------------------- training functions -------------

def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    total_prediction_num = y.shape[0]
    correct_prediction_num = (preds == y).nonzero().shape[0]
    return correct_prediction_num / total_prediction_num


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    :return A tuple of (average loss over all epoch's samples, accuracy over all epoch's examples)
    """
    # Initiate variables
    device = get_available_device()
    epoch_total_samples = 0
    epoch_average_loss = 0
    epoch_accuracy = 0

    # Start training model
    model.train()  # Move model mode to train
    for i, data in enumerate(data_iterator):
        sentences_embedded, sentences_real_sentiment = data[0].to(device), data[1].to(device)
        bach_size = sentences_embedded.shape[0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        sentences_logits = model(sentences_embedded).squeeze()
        bach_average_loss = criterion(sentences_logits, sentences_real_sentiment)
        bach_average_loss.backward()
        optimizer.step()

        # Updating epoch's average loss
        epoch_average_loss = (bach_average_loss.item() * bach_size + epoch_average_loss * epoch_total_samples) / \
                             (bach_size + epoch_total_samples)

        # Updating epoch's accuracy
        sentences_pred_sentiment = torch.round(torch.sigmoid(sentences_logits))
        batch_accuracy = binary_accuracy(sentences_pred_sentiment, sentences_real_sentiment)

        epoch_accuracy = (batch_accuracy * bach_size + epoch_accuracy * epoch_total_samples) / \
                         (bach_size + epoch_total_samples)

        # Update number of seen samples in the epoch
        epoch_total_samples += bach_size

    return epoch_average_loss, epoch_accuracy


def evaluate(model, data_iterator, criterion):
    """
    Evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: A tuple of (average loss over all epoch's samples, accuracy over all epoch's samples)
    """
    # Initiate variables
    device = get_available_device()
    data_total_samples = 0
    data_average_loss = 0
    data_accuracy = 0

    # Start evaluating model
    model.eval()  # Move model mode to test
    with torch.no_grad():
        for i, data in enumerate(data_iterator):
            sentences_embedded, sentences_real_sentiment = data[0].to(device), data[1].to(device)
            bach_size = sentences_embedded.shape[0]

            # Updating data's average loss
            sentences_logits = model(sentences_embedded).squeeze()
            bach_average_loss = criterion(sentences_logits, sentences_real_sentiment)
            data_average_loss = (bach_average_loss.item() * bach_size + data_average_loss * data_total_samples) / \
                                (bach_size + data_total_samples)

            # Updating data's accuracy
            sentences_pred_sentiment = torch.round(torch.sigmoid(sentences_logits))
            batch_accuracy = binary_accuracy(sentences_pred_sentiment, sentences_real_sentiment)
            data_accuracy = (batch_accuracy * bach_size + data_accuracy * data_total_samples) / \
                            (bach_size + data_total_samples)

            # Update number of seen samples
            data_total_samples += bach_size

    return data_average_loss, data_accuracy


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient computation
        for data in data_iter:
            inputs, _ = data
            inputs = inputs.to(get_available_device())
            preds = model.predict(inputs)
            predictions.extend(preds.cpu().numpy())  # Store predictions

    return np.array(predictions)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0., model_name=None):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    :param model_name: The name of the model
    :return: void
    """
    print("Training the model: {}".format(model_name))
    # Create a Binary Cross Entropy loss function that accepts logits and Adam optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initiate loss and accuracy arrays
    train_average_losses = []
    validation_average_losses = []

    train_accuracies = []
    validation_accuracies = []

    # Begin to train the model
    epochs = list(range(1, n_epochs + 1))
    for epoch in epochs:
        print("epoch number {}".format(epoch))
        # Train the model over one epoch and retrieve the average loss and accuracy
        model.train()
        train_iter = data_manager.get_torch_iterator(TRAIN)
        train_epoch_average_loss, train_epoch_accuracy = train_epoch(model=model, data_iterator=train_iter,
                                                                     optimizer=optimizer, criterion=criterion)

        train_average_losses.append(train_epoch_average_loss)
        train_accuracies.append(train_epoch_accuracy)

        # Evaluate the model over the validation samples and retrieve the average loss and accuracy
        val_iter = data_manager.get_torch_iterator(VAL)
        average_validation_loss, validation_accuracy = evaluate(model=model, data_iterator=val_iter, criterion=criterion)

        validation_average_losses.append(average_validation_loss)
        validation_accuracies.append(validation_accuracy)

    # Create and show requested plots
    create_plots(model_name, epochs, train_average_losses, validation_average_losses, train_accuracies, validation_accuracies)


# This method should create all the objects
# needed for the training process, and run the training process.
def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    :return: A tuple contain the trained Log_Linear_One_Hot model and its corresponding data manager
    """
    # Values specified in the exercise
    n_epochs = 20
    learning_rate = 1e-2
    weight_decay = 1e-3

    # Initiate DataManager instance
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                               batch_size=BATCH_SIZE)

    # Initialize Log Linear model and attach it to the device
    embedding_dim = len(data_manager.sent_func_kwargs['word_to_ind'].keys())
    model = LogLinear(embedding_dim)
    model.to(get_available_device())

    train_model(model_name="Log_Linear_One_Hot", model=model, data_manager=data_manager, n_epochs=n_epochs,
                lr=learning_rate, weight_decay=weight_decay)

    # Switch model's mode to evaluation and return the model
    model.eval()
    return model, data_manager


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    :return: A tuple contain the trained Log_Linear_W2V model and its corresponding data manager
    """
    # Values specified in the exercise
    n_epochs = 20
    learning_rate = 1e-2
    weight_decay = 1e-3

    # Initiate DataManager instance
    data_manager = DataManager(data_type=W2V_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                               batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)

    # Initialize Log Linear model and attach it to the device
    model = LogLinear(W2V_EMBEDDING_DIM)
    model.to(get_available_device())

    train_model(model_name="Log_Linear_W2V", model=model, data_manager=data_manager, n_epochs=n_epochs, lr=learning_rate,
                weight_decay=weight_decay)

    # Switch model's mode to evaluation and return the model
    model.eval()
    return model, data_manager


def train_lstm_with_w2v():
    """
    Train the LSTM model with words to embedding vectors
    :return: A tuple contain the trained LSTM_W2V model and its corresponding data manager
    """
    # Values specified in the exercise
    n_epochs = 4
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Initiate DataManager instance
    data_manager = DataManager(data_type=W2V_SEQUENCE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                               batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)

    # Initialize LSTM model and attach it to the device
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=HIDDEN_STATES_DIM, n_layers=1, dropout=0.5)
    model.to(get_available_device())

    # Train the LSTM model
    train_model(model_name="LSTM_W2V", model=model, data_manager=data_manager, n_epochs=n_epochs,
                lr=learning_rate, weight_decay=weight_decay)

    # Switch model's mode to evaluation and return the model
    model.eval()
    return model, data_manager


# Auxiliary functions
def create_plots(model_name, epochs, train_average_losses, validation_average_losses, train_accuracies, validation_accuracies):
    """
    Create and save two graphs: train and validation losses vs epoch number and train and validation accuracies vs epoch number
    :param model_name: The name if the model
    :param epochs: A List containing epoch numbers (to be use as domain)
    :param train_average_losses: A List containing average train loss for each epoch
    :param validation_average_losses: A List containing average validation loss for each epoch
    :param train_accuracies: A List containing train accuracy for each epoch
    :param validation_accuracies: A List containing validation accuracy for each epoch
    :return: void
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Initiate two subplots
    plot = make_subplots(rows=2, cols=1,
                         subplot_titles=("{}: Train and validation average loss as functions of epoch number".format(model_name),
                                         "{}: Train and validation accuracies as functions of epoch number".format(model_name)))

    # Create the first subplot to show average train and validation losses as functions of epoch number
    plot.update_xaxes(title_text="Epochs Number", row=1, col=1)
    plot.update_yaxes(title_text="Average Loss", row=1, col=1)
    plot.add_trace(
        go.Scatter(x=epochs, y=train_average_losses, mode='lines+markers', name='Average train loss',
                   marker=dict(color='#0000FF', size=10)), row=1, col=1
    )
    plot.add_trace(
        go.Scatter(x=epochs, y=validation_average_losses, mode='lines+markers', name='Average validation loss',
                   marker=dict(color='#8B0000', size=10)), row=1, col=1
    )

    # Create the second subplot to show train and validation train accuracies as functions of epoch number
    plot.update_xaxes(title_text="Epochs Number", row=2, col=1)
    plot.update_yaxes(title_text="Accuracy", row=2, col=1)
    plot.add_trace(
        go.Scatter(x=epochs, y=train_accuracies, mode='lines+markers', name='Train accuracy',
                   marker=dict(color='#0000FF', size=10)), row=2, col=1
    )
    plot.add_trace(
        go.Scatter(x=epochs, y=validation_accuracies, mode='lines+markers', name='Validation accuracy',
                   marker=dict(color='#8B0000', size=10)), row=2, col=1
    )

    plot.write_image('losses_and_accuracies_for_{}.png'.format(model_name))
    plot.show()


def train_and_test_model(model_name):
    """
    This function warp the hole training, validation and testing process of the asked model.
    In addition, the model's accuracy is tested against special database's subsets. All results are saved in the file results.log
    :param model_name: The name of the asked model (supporting 'Log_Linear_One_Hot', 'Log_Linear_W2V', and 'LSTM_W2V')
    :return: Void
    """
    # Initiate setting to print and save the results to a log file
    import logging
    logging.basicConfig(filename='results.log', filemode='w', format='[%(funcName)s]  %(message)s', level=logging.INFO)

    logging.info("Got the model name: '{}'\n".format(model_name))
    if model_name not in ['Log_Linear_One_Hot', 'Log_Linear_W2V', 'LSTM_W2V']:
        logging.info("Error - the model name '{}' is not supported. Exit.".format(model_name))
        return

    logging.info("Create {} model and start to train it against the training dataset".format(model_name))
    if model_name == 'Log_Linear_One_Hot':
        model, data_manager = train_log_linear_with_one_hot()
    elif model_name == 'Log_Linear_W2V':
        model, data_manager = train_log_linear_with_w2v()
    else:  # model_name == 'LSTM_W2V'
        model, data_manager = train_lstm_with_w2v()

    logging.info("Test the {} model against the validation dataset:".format(model_name))
    criterion = nn.BCEWithLogitsLoss()
    val_iter = data_manager.get_torch_iterator(VAL)
    average_val_loss, val_accuracy = evaluate(model=model, data_iterator=val_iter, criterion=criterion)
    logging.info("{} model's average loss: {}".format(model_name, average_val_loss))
    logging.info("{} model's accuracy: {}\n".format(model_name, val_accuracy))

    logging.info("Test the {} model against the test dataset:".format(model_name))
    criterion = nn.BCEWithLogitsLoss()
    test_iter = data_manager.get_torch_iterator(TEST)
    average_test_loss, test_accuracy = evaluate(model=model, data_iterator=test_iter, criterion=criterion)
    logging.info("{} model's average loss: {}".format(model_name, average_test_loss))
    logging.info("{} model's accuracy: {}\n".format(model_name, test_accuracy))

    logging.info("Test the {} model against sentences with negative polarity:".format(model_name))
    negative_polarity_iter = data_manager.get_torch_iterator(NEGATED_POLARITY)
    _, negative_polarity_accuracy = evaluate(model=model, data_iterator=negative_polarity_iter, criterion=criterion)
    logging.info("{} model's accuracy: {}\n".format(model_name, negative_polarity_accuracy))

    logging.info("Test the {} model against sentences with rare words:".format(model_name))
    rare_words_iter = data_manager.get_torch_iterator(RARE_WORDS)
    _, rare_words_accuracy = evaluate(model=model, data_iterator=rare_words_iter, criterion=criterion)
    logging.info("{} model's accuracy: {}\n\n".format(model_name, rare_words_accuracy))


if __name__ == '__main__':
    # Question 6
    train_and_test_model('Log_Linear_One_Hot')
    # Question 7
    train_and_test_model('Log_Linear_W2V')
    # Question 8
    train_and_test_model('LSTM_W2V')
