#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Att_BLSTM(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        
        # Initialize word embeddings, class number, and configuration
        self.word_vec = word_vec  # Pre-trained word embeddings
        self.class_num = class_num  # Number of output classes for classification

        # Hyperparameters and configurations
        self.max_len = config.max_len  # Maximum sequence length
        self.word_dim = config.word_dim  # Word embedding dimension
        self.hidden_size = config.hidden_size  # Hidden size for LSTM
        self.layers_num = config.layers_num  # Number of LSTM layers
        self.emb_dropout_value = config.emb_dropout  # Dropout rate for embedding
        self.lstm_dropout_value = config.lstm_dropout  # Dropout rate for LSTM
        self.linear_dropout_value = config.linear_dropout  # Dropout rate for linear layer

        # Define the model structure

        # Word embedding layer initialized with pre-trained word vectors
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,  # Load pre-trained embeddings
            freeze=False,  # Allow fine-tuning the embeddings during training
        )

        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.word_dim,  # Input size is the word embedding dimension
            hidden_size=self.hidden_size,  # Hidden state size
            num_layers=self.layers_num,  # Number of LSTM layers
            bias=True,  # Use bias terms
            batch_first=True,  # The input batch size is the first dimension
            dropout=0,  # No dropout between LSTM layers
            bidirectional=True,  # Use a bidirectional LSTM
        )

        # Tanh activation function
        self.tanh = nn.Tanh()

        # Dropout layers for regularization
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        # Attention mechanism weight
        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))

        # Fully connected (linear) layer for classification
        self.dense = nn.Linear(
            in_features=self.hidden_size,  # The input feature size is the hidden size of the LSTM
            out_features=self.class_num,  # The output size is the number of classes
            bias=True  # Use bias term
        )

        # Initialize the weights of the dense layer using Xavier normal initialization
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        """
        Process input sequences with a BiLSTM layer, handling padded sequences with mask.
        """
        lengths = torch.sum(mask.gt(0), dim=-1)  # Calculate lengths of sequences (ignoring padding)
        lengths = lengths.cpu()  # Move lengths to CPU (to avoid potential errors)
        
        # Pack padded sequence before feeding into LSTM (efficient processing of variable length sequences)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through the LSTM layer
        h, (_, _) = self.lstm(x)  # h: hidden states from all time steps, (_, _): ignore cell state
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)  # Unpack and pad sequences
        
        # Reshape the output for further processing (summarizing bidirectional output)
        h = h.view(-1, self.max_len, 2, self.hidden_size)  # B*L*2H, where B=Batch, L=Sequence Length, H=Hidden Size
        h = torch.sum(h, dim=2)  # Summing bidirectional outputs to get B*L*H
        return h

    def attention_layer(self, h, mask):
        """
        Apply attention mechanism on the output of the LSTM layer to focus on important parts of the sequence.
        """
        # Expand attention weights to match batch size
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # B*H*1
        
        # Calculate attention scores using the Tanh activation
        att_score = torch.bmm(self.tanh(h), att_weight)  # B*L*H * B*H*1 -> B*L*1

        # Mask the attention scores to ignore padding positions
        mask = mask.unsqueeze(dim=-1)  # B*L*1 (Adding a dimension to the mask)
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # B*L*1 (Set masked positions to -inf)
        
        # Apply softmax to get attention weights
        att_weight = F.softmax(att_score, dim=1)  # B*L*1 (Normalize the attention scores)

        # Calculate the context vector (weighted sum of hidden states)
        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L * B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps)  # Apply Tanh activation on the context vector
        return reps

    def forward(self, data):
        """
        Forward pass for the model.
        """
        # Extract token and mask from the input data (assumed to be a tensor of shape (B, 2, L))
        token = data[:, 0, :].view(-1, self.max_len)  # Tokenized word indices (B*L)
        mask = data[:, 1, :].view(-1, self.max_len)   # Mask for padding positions (B*L)
        
        # Apply embedding layer
        emb = self.word_embedding(token)  # B*L*word_dim
        emb = self.emb_dropout(emb)  # Apply dropout to embeddings

        # Pass through LSTM layer
        h = self.lstm_layer(emb, mask)  # B*L*H (LSTM hidden states)
        h = self.lstm_dropout(h)  # Apply dropout to LSTM output

        # Apply attention mechanism to get context vector
        reps = self.attention_layer(h, mask)  # B*reps (context vector)

        # Apply dropout to the context vector before classification
        reps = self.linear_dropout(reps)

        # Pass through the final fully connected layer (dense layer) for classification
        logits = self.dense(reps)  # Output logits (B*class_num)

        return logits  # Return the logits for classification
