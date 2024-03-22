

###################################################
# Exercise 4 - Natural Language Processing 67658  #
###################################################


import numpy as np

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param categories: The selected cagiest we want to filter out of the database
    :param portion: Portion of the data to use for training
    :return: The train and test data
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()

    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Internal function
    def convert_to_tfidf_embedding(x_train, x_test):
        """
        :param x_train: The train features
        :param x_test: The test features
        :return: The tfidf embedding vectors
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(stop_words='english', max_features=1000)
        x_train = tf.fit_transform(x_train).toarray()
        x_test = tf.transform(x_test).toarray()

        return x_train, x_test

    # Load dataset and convert to TFIDF embedding vectors
    X_train, y_train, X_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    X_train, X_test = convert_to_tfidf_embedding(X_train, X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Initiate a Logistic Regression model, train the model over the train dataset and inference over the test dataset
    logistic_reg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)
    y_pred = logistic_reg_model.predict(X_test)

    # Compute and return the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch
    import logging
    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Internal class
    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # Internal function
    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initiate tokenizer and auto model
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    # Load train and evaluation databases and create corresponding datasets
    X_train, y_train, X_eval, y_eval = get_data(categories=category_dict.keys(), portion=portion)

    train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    train_encodings = {key: val.numpy().tolist() for key, val in train_encodings.items()}
    train_dataset = Dataset(encodings=train_encodings, labels=y_train)

    eval_encodings = tokenizer(X_eval, padding=True, truncation=True, return_tensors="pt")
    eval_encodings = {key: val.numpy().tolist() for key, val in eval_encodings.items()}
    eval_dataset = Dataset(encodings=eval_encodings, labels=y_eval)

    # Create Trainer object to train the model
    learning_rate = 5e-5
    batch_size = 16
    num_of_epochs = 3

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_of_epochs,
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        output_dir='log'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Begin to train the model (fine-tuning)
    trainer.train()

    # Report train average loss and evaluation accuracy of each epoch
    print(f"\nPortion {portion} - train average loss and evaluation accuracy of each epoch:")
    for epoch in range(1, num_of_epochs + 1):
        average_training_loss = None
        evaluation_accuracy = None

        for log in trainer.state.log_history:
            if 'epoch' in log.keys() and log['epoch'] == epoch:
                if 'loss' in log.keys():
                    average_training_loss = log['loss']
                if 'eval_accuracy' in log.keys():
                    evaluation_accuracy = log['eval_accuracy']

        if average_training_loss is not None and evaluation_accuracy is not None:
            print(
                f"Epoch {epoch} - Average Training Loss: {average_training_loss:.4f}, "
                f"Validation Accuracy: {evaluation_accuracy:.4f}"
            )
    print("\n")

    # Evaluate the model' accuracy on the evaluation dataset
    accuracy = trainer.evaluate(eval_dataset=eval_dataset)['eval_accuracy']
    return accuracy


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch

    # Lode the tests dataset
    _, _, X_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Create a zero-shot pipline
    device = 0 if torch.cuda.is_available() else -1
    batch_size = 16
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768', device=device,
                   batch_size=batch_size)

    # Begin inference over the test dataset
    # Split test data to batch of 16 to not overloaded the memory and speed-up the inference process
    pred_test = []
    candidate_labels = list(category_dict.values())

    for i in range(0, len(X_test), 16):
        X_batch = X_test[i: min(i + batch_size, len(X_test))]
        results = clf(X_batch, candidate_labels=candidate_labels)
        batch_predictions = [candidate_labels.index(result['labels'][0]) for result in results]
        pred_test.extend(batch_predictions)

    # Compute and return the accuracy
    accuracy = accuracy_score(y_test, pred_test)
    return accuracy


# Auxiliary function
def create_plot(model_name, data_portions, accuracies):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Initiate a subplots
    plot = make_subplots(rows=1, cols=1)

    # Update the subplot to show test accuracy as functions of portion of test dataset
    plot.update_xaxes(title_text="Portion of train dataset", row=1, col=1)
    plot.update_yaxes(title_text="Test accuracy", row=1, col=1)
    plot.add_trace(
        go.Scatter(x=data_portions, y=accuracies, mode='lines+markers', name='Test accuracy',
                   marker=dict(color='#0000FF', size=10)), row=1, col=1
    )

    plot.update_layout(title_text="{}: Test accuracy as a function of the portion of the train dataset".format(model_name))
    plot.write_image('test_accuracy_for_{}.png'.format(model_name))
    plot.show()


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]

    # Q1
    print("Logistic regression results:")
    lr_accuracies = []
    for p in portions:
        accuracy = linear_classification(portion=p)
        lr_accuracies.append(accuracy)
        print(f"\nPortion: {p}")
        print(f"{accuracy:.4f}")

    create_plot(model_name="Logistic regression", data_portions=portions, accuracies=lr_accuracies)

    # Q2
    print("\nFine-tuning results:")
    tran_accuracies = []
    for p in portions:
        accuracy = transformer_classification(portion=p)
        tran_accuracies.append(accuracy)
        print(f"\nPortion: {p}")
        print(f"{accuracy:.4f}")

    create_plot(model_name="Fine-tuned transformer", data_portions=portions, accuracies=tran_accuracies)

    # Q3
    print("\nZero-shot result:")
    print(f"{zeroshot_classification():.4f}")

