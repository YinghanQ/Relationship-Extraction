#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.8

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Class to load pre-trained word embeddings
class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embeddings.
    It loads the embeddings from a specified file and processes them into a usable format.
    """

    def __init__(self, config):
        # Initialize with configuration that contains the file path and word dimension
        self.path_word = config.embedding_path  # path to pre-trained word embeddings
        self.word_dim = config.word_dim  # dimension of the word embeddings

    def load_embedding(self):
        """
        Loads word embeddings from a file and processes them into a dictionary format.
        
        Returns:
            word2id (dict): A dictionary mapping words to unique indices.
            word_vec (torch.Tensor): A tensor containing word embeddings.
        """
        word2id = dict()  # Map to store word to index
        word_vec = list()  # List to store word embeddings corresponding to each word index

        # Special tokens: PAD, UNK, <e1>, <e2>, </e1>, </e2>
        word2id['PAD'] = len(word2id)  # Index for padding token
        word2id['UNK'] = len(word2id)  # Index for unknown token
        word2id['<e1>'] = len(word2id)
        word2id['<e2>'] = len(word2id)
        word2id['</e1>'] = len(word2id)
        word2id['</e2>'] = len(word2id)

        # Open the embedding file and read the word vectors
        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()  # Split the line into word and its embedding components
                if len(line) != self.word_dim + 1:
                    continue  # Skip lines that don't match the expected format
                word2id[line[0]] = len(word2id)  # Add word to word2id mapping
                word_vec.append(np.asarray(line[1:], dtype=np.float32))  # Store word embedding as a numpy array

        word_vec = np.stack(word_vec)  # Convert list of embeddings into a numpy array
        vec_mean, vec_std = word_vec.mean(), word_vec.std()  # Calculate mean and std of word embeddings

        # Initialize embeddings for special tokens
        special_emb = np.random.normal(vec_mean, vec_std, (6, self.word_dim))  # Random embeddings for special tokens
        special_emb[0] = 0  # <pad> embedding is initialized as zero

        # Concatenate the special token embeddings with the regular word embeddings
        word_vec = np.concatenate((special_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)  # Reshape to ensure correct dimensions
        word_vec = torch.from_numpy(word_vec)  # Convert numpy array to PyTorch tensor
        return word2id, word_vec  # Return the word-to-id mapping and word embeddings as a tensor


# Class to load relation mappings
class RelationLoader(object):
    """
    A loader for relations in a dataset.
    It loads the relation-to-id and id-to-relation mappings from a file.
    """
    def __init__(self, config):
        self.data_dir = config.data_dir  # Directory where relation file is stored

    def __load_relation(self):
        """
        Loads the relation mappings from a file into dictionaries.
        
        Returns:
            rel2id (dict): A dictionary mapping relation strings to unique indices.
            id2rel (dict): A dictionary mapping indices to relation strings.
            num_relations (int): The number of unique relations.
        """
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')  # Path to relation mapping file
        rel2id = {}  # Map relation to id
        id2rel = {}  # Map id to relation
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()  # Split the line into relation and its ID
                id_d = int(id_s)  # Convert ID to integer
                rel2id[relation] = id_d  # Add relation to rel2id mapping
                id2rel[id_d] = relation  # Add id to id2rel mapping
        return rel2id, id2rel, len(rel2id)  # Return the relation mappings and the total number of relations

    def get_relation(self):
        """
        Returns the relation mappings (rel2id, id2rel).
        """
        return self.__load_relation()


# Custom Dataset class to handle the SemEval dataset
class SemEvalDateset(Dataset):
    """
    Dataset class for SemEval data.
    It processes and loads sentences and their corresponding relations.
    """
    def __init__(self, filename, rel2id, word2id, config):
        self.filename = filename  # File containing the data
        self.rel2id = rel2id  # Relation to ID mapping
        self.word2id = word2id  # Word to ID mapping
        self.max_len = config.max_len  # Maximum sentence length (for padding)
        self.data_dir = config.data_dir  # Data directory
        self.dataset, self.label = self.__load_data()  # Load dataset and labels

    def __symbolize_sentence(self, sentence):
        """
        Convert a sentence into a symbolic format with word IDs and a mask for padding.

        Args:
            sentence (list): A list of words in the sentence.

        Returns:
            unit (numpy.ndarray): A 2D array where each row contains word IDs and the corresponding mask.
        """
        mask = [1] * len(sentence)  # Initially mark all words as valid (1)
        words = []  # List to store word IDs
        length = min(self.max_len, len(sentence))  # Limit the sentence length to max_len
        mask = mask[:length]  # Truncate the mask if sentence is too long

        # Convert words to their corresponding IDs, using 'UNK' for unknown words
        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))

        # Pad the sentence to the max length
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # Add 0 in the mask for padding tokens
                words.append(self.word2id['PAD'])  # Add padding token to the word list

        # Return the words and mask as a numpy array, reshaped to fit model requirements
        unit = np.asarray([words, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 2, self.max_len))
        return unit

    def __load_data(self):
        """
        Loads data from a file, processes each sentence, and stores it along with its label.

        Returns:
            data (list): A list of sentences represented by word IDs and masks.
            labels (list): A list of integer labels corresponding to each sentence's relation.
        """
        path_data_file = os.path.join(self.data_dir, self.filename)  # Path to data file
        data = []  # List to store processed data
        labels = []  # List to store labels
        with open(path_data_file, 'r', encoding='utf-8-sig') as fr:
            for line in fr:
                line = json.loads(line.strip())  # Parse the JSON line
                label = line['relation']  # Extract the relation label
                sentence = line['sentence']  # Extract the sentence
                label_idx = self.rel2id[label]  # Get the corresponding label index
    
                one_sentence = self.__symbolize_sentence(sentence)  # Process the sentence
                data.append(one_sentence)  # Add processed sentence to data
                labels.append(label_idx)  # Add corresponding label to labels
        return data, labels  # Return the processed data and labels

    def __getitem__(self, index):
        """
        Get a data sample and its corresponding label by index.

        Args:
            index (int): Index of the data sample.

        Returns:
            data (torch.Tensor): A tensor containing the word IDs and mask.
            label (int): The integer label corresponding to the relation.
        """
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        """
        Returns the total number of data samples in the dataset.
        """
        return len(self.label)


# DataLoader class to handle batching and loading of the SemEval dataset
class SemEvalDataLoader(object):
    """
    A custom data loader that handles batching, shuffling, and loading for SemEval dataset.
    """
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id  # Relation to ID mapping
        self.word2id = word2id  # Word to ID mapping
        self.config = config  # Configuration containing batch size and other parameters

    def __collate_fn(self, batch):
        """
        Custom collate function to handle batch processing.
        It combines data and labels from the batch into tensors.

        Args:
            batch (list): A list of tuples, each containing a data sample and its label.

        Returns:
            data (torch.Tensor): A tensor containing all data samples in the batch.
            label (torch.Tensor): A tensor containing all labels in the batch.
        """
        data, label = zip(*batch)  # Unzip the batch into data and labels
        data = list(data)  # Convert to list
        label = list(label)  # Convert to list
        data = torch.from_numpy(np.concatenate(data, axis=0))  # Concatenate and convert to tensor
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))  # Convert labels to tensor
        return data, label

    def __get_data(self, filename, shuffle=False):
        """
        Load a dataset from a file and return a DataLoader instance.

        Args:
            filename (str): The name of the dataset file.
            shuffle (bool): Whether to shuffle the data during loading.

        Returns:
            loader (DataLoader): A DataLoader instance for batching and loading data.
        """
        dataset = SemEvalDateset(filename, self.rel2id, self.word2id, self.config)  # Load dataset
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,  # Set batch size from config
            shuffle=shuffle,  # Shuffle data if needed
            num_workers=2,  # Set number of workers for parallel loading
            collate_fn=self.__collate_fn  # Use custom collate function
        )
        return loader

    def get_train(self):
        """
        Get the DataLoader for the training dataset.
        """
        return self.__get_data('train.json', shuffle=True)

    def get_dev(self):
        """
        Get the DataLoader for the development (validation) dataset.
        """
        return self.__get_data('validation.json', shuffle=False)

    def get_test(self):
        """
        Get the DataLoader for the test dataset.
        """
        return self.__get_data('test.json', shuffle=False)


# Main script to load data and test the loader
if __name__ == '__main__':
    from config import Config
    config = Config()  # Load configuration
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()  # Load word embeddings
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()  # Load relation mappings
    loader = SemEvalDataLoader(rel2id, word2id, config)  # Initialize DataLoader
    test_loader = loader.get_train()  # Get the training data loader

    # Test data loading
    for step, (data, label) in enumerate(test_loader):
        print(type(data), data.shape)  # Print data type and shape
        print(type(label), label.shape)  # Print label type and shape
        break  # Exit after the first batch to avoid excessive printing
