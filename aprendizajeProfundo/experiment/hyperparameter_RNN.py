import argparse
import gzip
import json
import logging
import mlflow
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm, trange

import numpy as np

from functools import partial

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

from .utils import PadSequences
from .dataset import MeLiChallengeDataset


logging.basicConfig(format='%(asctime)s: %(levelname)s - %(message)s', level=logging.INFO)


class RNNClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 hidden_layers,
                 dropout,
                 lstm_features,
                 lstm_layers,
                 vector_size,
                 freeze_embedings):
        super().__init__()
        with gzip.open(token_to_index, 'rt') as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, 'rt') as fh:
            # Let's fill the embedding matrix
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    wordID = token_to_index[word]
                    embeddings_matrix[wordID] = torch.FloatTensor([float(n) for n in vector.split()])
        # Embedding Layer
        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        # LSTM Layers
        self.lstm_config = {'input_size': vector_size,
                            'hidden_size': lstm_features,
                            'num_layers': lstm_layers,
                            'batch_first': True,
                            'dropout': dropout if lstm_layers > 1 else 0.0}
        self.lstm = nn.LSTM(**self.lstm_config)
        # Linear Layers
        self.hidden_layers = [
            nn.Linear(lstm_features, hidden_layers[0])
        ]
        for input_size, output_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(
                nn.Linear(input_size, output_size)
            )
        self.dropout = dropout
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(hidden_layers[-1], n_labels)
        self.vector_size = vector_size

    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.lstm(x)
        # Take last state of lstm, which is a representation of the entire text
        x = x[:, -1, :].squeeze()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.dropout:
                x = F.dropout(x, self.dropout)
        x = self.output(x)
        return x


def train_val_test_model(train_data,
                         validation_data,
                         test_data,
                         token_to_index,
                         pretrained_embeddings,
                         language,
                         embeddings_size,
                         hidden_layers,
                         dropout,
                         lstm_features,
                         lstm_layers,
                         epochs,
                         batch_size,
                         freeze_embeddings,
                         learning_rate,
                         weight_decay,
                         random_buffer_size=2048):
    # Padding
    pad_sequences = PadSequences(
        pad_value=0,
        max_length=None,
        min_length=1
    )
    # Train Dataset
    logging.info('Building training dataset')
    train_dataset = MeLiChallengeDataset(
        dataset_path=train_data,
        random_buffer_size=random_buffer_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )
    # Val Dataset
    logging.info('Building validation dataset')
    validation_dataset = MeLiChallengeDataset(
        dataset_path=validation_data,
        random_buffer_size=1
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )
    # Test Dataset
    if test_data is not None:
        logging.info('Building test dataset')
        test_dataset = MeLiChallengeDataset(
            dataset_path=test_data,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    # Start experiment!
    mlflow.set_experiment(f'DiploDatos.{language}')

    with mlflow.start_run():
        logging.info('Starting experiment')
        # Log all relevent hyperparameters
        mlflow.log_params({
            'model_type': 'Hyperparameter RNN',
            'embeddings': pretrained_embeddings,
            'hidden_layers': hidden_layers,
            'dropout': dropout,
            'lstm_features': lstm_features,
            'lstm_layers': lstm_layers,
            'embeddings_size': embeddings_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'freeze_embeddings': freeze_embeddings,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'random_buffer_size': random_buffer_size   
        })
        # Look for the device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        logging.info('Building classifier')
        model = RNNClassifier(
                pretrained_embeddings_path=pretrained_embeddings,
                token_to_index=token_to_index,
                n_labels=train_dataset.n_labels,
                hidden_layers=hidden_layers,
                dropout=dropout,
                lstm_features=lstm_features,
                lstm_layers=lstm_layers,
                vector_size=embeddings_size,
                freeze_embedings=freeze_embeddings
                )
        # Send the model to the device
        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        main_metric = []
        logging.info('Training classifier')
        for epoch in trange(epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch['data'].to(device)
                target = batch['target'].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric('train_loss', sum(running_loss) / len(running_loss), epoch)

            logging.info('Evaluating model on validation')
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(validation_loader):
                    data = batch['data'].to(device)
                    target = batch['target'].to(device)
                    output = model(data)
                    loss_value = loss(output, target)
                    running_loss.append(loss_value.item())
                    targets.extend(batch['target'].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric('validation_loss', sum(running_loss) / len(running_loss), epoch)
                balanced_Acc = balanced_accuracy_score(targets, predictions)
                mlflow.log_metric('validation_bacc', balanced_Acc, epoch)
                main_metric.append(balanced_Acc)

        if test_dataset:
            logging.info('Evaluating model on test')
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch['data'].to(device)
                    target = batch['target'].to(device)
                    output = model(data)
                    loss_value = loss(output, target)
                    running_loss.append(loss_value.item())
                    targets.extend(batch['target'].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric('test_loss', sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric('test_bacc', balanced_accuracy_score(targets, predictions), epoch)
            # This is the best model that we found!
            return model
    # This is the metric that we want to maximize!
    return main_metric


# Define an objective function
def objective_function(# Searched Parameters
                       searched_params,
                       # Fixed Parameters
                       train_data,
                       validation_data,
                       test_data,
                       token_to_index,
                       pretrained_embeddings,
                       language,
                       embeddings_size):
    logging.info('Exploring config...')

    main_metric = train_val_test_model(
                  train_data,
                  validation_data,
                  test_data,
                  token_to_index,
                  pretrained_embeddings,
                  language,
                  embeddings_size,
                  **searched_params)

    history = {}
    # This is the value that will be minimized!
    history['loss'] = main_metric[-1] * -1
    # This is a required key
    history['status'] = STATUS_OK

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Train Path
    parser.add_argument('--train-data',
                        help='Path to the training dataset.',
                        required=True)
    # Validation Path
    parser.add_argument('--validation-data',
                        help='Path to the validation dataset.',
                        required=True)
    # Test Path
    parser.add_argument('--test-data',
                        help='Path to the test dataset.',
                        required=True)
    # Token to Index Path
    parser.add_argument('--token-to-index',
                        help='Path to the json file that maps tokens to indices.',
                        required=True)
    # Pretrained Embeddings Path
    parser.add_argument('--pretrained-embeddings',
                        help='Path to the pretrained embeddings file.',
                        required=True)
    # Language
    parser.add_argument('--language',
                        help='Language working with.',
                        required=True)
    # Embedding Size
    parser.add_argument('--embeddings-size',
                        default=300,
                        help='Size of the vectors.',
                        type=int)
    args = parser.parse_args()

    # Some parameteres never change...
    fixed_params = {'train_data': args.train_data,
                    'validation_data': args.validation_data,
                    'token_to_index': args.token_to_index,
                    'pretrained_embeddings': args.pretrained_embeddings,
                    'language': args.language,
                    'embeddings_size': args.embeddings_size
                   }

    # Define the search space...
    hidden_layers_choices = [[128],
                             [256, 128],
                             [256, 514, 256]]
    epochs_choices = [1, 3, 5, 10, 15]
    batch_size_choices = [128, 256, 514]
    freeze_embeddings_choices = [True, False]
    lstm_features_choices = [128, 256, 514]
    lstm_layers_choices = [1, 3, 5]
    space = {'hidden_layers': hp.choice('hidden_layers', hidden_layers_choices),
             'epochs': hp.choice('epochs', epochs_choices),
             'batch_size': hp.choice('batch_size', batch_size_choices),
             'freeze_embeddings': hp.choice('freeze_embeddings', freeze_embeddings_choices),
             'lstm_features': hp.choice('lstm_features', lstm_features_choices),
             'lstm_layers': hp.choice('lstm_layers', lstm_layers_choices),
             'dropout': hp.uniform('dropout', 0, 0.5),
             'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
             'weight_decay': hp.loguniform('weight_decay', np.log(0.00001), np.log(0.1))
            }

    # Define the Trials object, which will allow us to store information from every experiment.
    trials = Trials()

    partial_params = {**fixed_params, 'test_data': None}
    # We define some partial function to minimize
    fmin_objective = partial(objective_function, **partial_params)

    # Minimize the objective function over the space
    best_params = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
    # We need to get back the proper values selected
    best_params['hidden_layers'] = hidden_layers_choices[best_params['hidden_layers']]
    best_params['epochs'] = epochs_choices[best_params['epochs']]
    best_params['batch_size'] = batch_size_choices[best_params['batch_size']]
    best_params['freeze_embeddings'] = freeze_embeddings_choices[best_params['freeze_embeddings']]
    best_params['lstm_features'] = lstm_features_choices[best_params['lstm_features']]
    best_params['lstm_layers'] = lstm_layers_choices[best_params['lstm_layers']]

    # Get the final results :)
    full_params = {**fixed_params, **best_params, 'test_data': args.test_data}

    best_model = train_val_test_model(**full_params)
    # We could do something with this model if we wanted...
