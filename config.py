#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.8


import argparse  # Used for parsing command-line arguments.
import torch  # PyTorch library for deep learning.
import os  # Provides functions to interact with the operating system.
import random  # Used for generating random numbers.
import json  # Library for working with JSON data.
import numpy as np  # NumPy library for numerical computing.

class Config(object):
    def __init__(self):
        # Get initial configuration from command-line arguments.
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # Select computing device (GPU or CPU).
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))  # Use specified GPU.
        else:
            self.device = torch.device('cpu')  # Default to CPU if GPU is unavailable.

        # Determine the model name and create a directory to save model files.
        if self.model_name is None:
            self.model_name = 'Att_BLSTM_CNN'  # Default model name.
        self.model_dir = os.path.join(self.output_dir, self.model_name)  # Model save path.
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)  # Create the directory if it does not exist.

        # Backup configuration settings.
        self.__config_backup(args)

        # Set random seed for reproducibility.
        self.__set_seed(self.seed)

    def __get_config(self):
        # Define and parse command-line arguments.
        parser = argparse.ArgumentParser()
        parser.description = 'Configuration for models'

        # Paths for data and output.
        parser.add_argument('--data_dir', type=str,
                            default='./data',
                            help='Directory to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='Directory to save output')

        # Word embedding settings.
        parser.add_argument('--embedding_path', type=str,
                            default='./embedding/glove.6B.100d.txt',
                            help='Path to pre-trained word embedding file')
        parser.add_argument('--word_dim', type=int,
                            default=100,
                            help='Dimension of word embeddings')

        # CNN settings.
        parser.add_argument('--cnn_filters', type=int,
                            default=128,
                            help='Number of output channels for the CNN layer')

        # Training settings.
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='Name of the model')
        parser.add_argument('--mode', type=int,
                            default=1,
                            choices=[0, 1],
                            help='Running mode: 1 for training, 0 for testing')
        parser.add_argument('--seed', type=int,
                            default=5782,
                            help='Random seed for reproducibility')
        parser.add_argument('--cuda', type=int,
                            default=0,
                            help='GPU device number; -1 for CPU')
        parser.add_argument('--epoch', type=int,
                            default=20,
                            help='Maximum number of training epochs')

        # Hyperparameters.
        parser.add_argument('--batch_size', type=int,
                            default=10,
                            help='Batch size for training')
        parser.add_argument('--lr', type=float,
                            default=1.0,
                            help='Learning rate for optimization')
        parser.add_argument('--max_len', type=int,
                            default=100,
                            help='Maximum sentence length')

        # Dropout rates.
        parser.add_argument('--emb_dropout', type=float,
                            default=0.3,
                            help='Dropout probability for the embedding layer')
        parser.add_argument('--lstm_dropout', type=float,
                            default=0.3,
                            help='Dropout probability for the (Bi)LSTM layer')
        parser.add_argument('--linear_dropout', type=float,
                            default=0.5,
                            help='Dropout probability for the linear layer')

        # LSTM settings.
        parser.add_argument('--hidden_size', type=int,
                            default=100,
                            help='Dimension of hidden units in the (Bi)LSTM layer')
        parser.add_argument('--layers_num', type=int,
                            default=1,
                            help='Number of RNN layers')

        # Regularization settings.
        parser.add_argument('--L2_decay', type=float, default=1e-5,
                            help='L2 weight decay for regularization')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        """Set random seed for reproducibility across different libraries."""
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)  # Set Python's hash seed.
        random.seed(seed)  # Set seed for Python's built-in random module.
        np.random.seed(seed)  # Set seed for NumPy.
        torch.manual_seed(seed)  # Set seed for PyTorch on CPU.
        torch.cuda.manual_seed(seed)  # Set seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  # Set seed for all available GPUs.

    def __config_backup(self, args):
        """Save configuration settings as a JSON file."""
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)  # Save settings in JSON format.

    def print_config(self):
        """Print all configuration settings."""
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])

if __name__ == '__main__':
    # Initialize configuration and print settings.
    config = Config()
    config.print_config()
