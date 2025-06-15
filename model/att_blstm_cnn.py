#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Define the model class with BiLSTM, Attention, and CNN layers
class Att_BLSTM_CNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec  # Pre-trained word embeddings
        self.class_num = class_num  # Number of output classes

        # Hyperparameters from the config
        self.max_len = config.max_len  # Maximum sequence length
        self.word_dim = config.word_dim  # Dimension of word embeddings
        self.hidden_size = config.hidden_size  # Hidden size of LSTM
        self.layers_num = config.layers_num  # Number of LSTM layers
        self.emb_dropout_value = config.emb_dropout  # Dropout for embedding layer
        self.lstm_dropout_value = config.lstm_dropout  # Dropout for LSTM layer
        self.linear_dropout_value = config.linear_dropout  # Dropout for fully connected layer
        self.cnn_filters = config.cnn_filters  # Number of filters in the CNN layers

        # Embedding Layer: Using pre-trained word embeddings for initialization
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,  # Whether to fine-tune the word embeddings
        )

        # BiLSTM Layer: Bidirectional LSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=self.word_dim,  # Input dimension (word embedding dimension)
            hidden_size=self.hidden_size,  # Hidden state size
            num_layers=self.layers_num,  # Number of LSTM layers
            bias=True,  # Whether to include bias terms
            batch_first=True,  # Batch comes first in the input tensor
            dropout=0,  # No dropout between LSTM layers
            bidirectional=True,  # Bidirectional LSTM
        )

        # CNN Layer: Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.hidden_size, 
                      out_channels=self.cnn_filters, 
                      kernel_size=k, 
                      padding=k // 2) for k in [2, 3, 4]  # Kernel sizes 2, 3, and 4
        ])

        # Attention Layer: To compute a weighted sum of LSTM outputs
        self.tanh = nn.Tanh()  # Non-linear activation for attention scores
        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))  # Attention weights

        # Dropout Layers: Dropout for regularization
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        # Fully Connected Layer: Output layer for classification
        total_features = self.hidden_size + self.cnn_filters * len(self.convs)
        self.dense = nn.Linear(total_features, self.class_num, bias=True)  # Final dense layer for classification

        # Weight Initialization: Xavier initialization for weights
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    # LSTM Layer: Process the sequence using BiLSTM
    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)  # Calculate sequence lengths
        lengths = lengths.cpu()  # Ensure lengths are on CPU to avoid errors
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)  # Pack padded sequences
        h, (_, _) = self.lstm(x)  # Run through BiLSTM
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)  # Pad sequences back
        h = h.view(-1, self.max_len, 2, self.hidden_size)  # Reshape to include bidirectional outputs
        h = torch.sum(h, dim=2)  # Sum bidirectional outputs (forward + backward)
        return h

    # Attention Layer: Apply attention mechanism on the BiLSTM output
    def attention_layer(self, h, mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # Expand attention weight to match batch size
        att_score = torch.bmm(self.tanh(h), att_weight)  # Compute attention scores for each timestep

        # Masking to ignore padding positions
        mask = mask.unsqueeze(dim=-1)  # Add an extra dimension to mask
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # Mask padding positions with negative infinity
        att_weight = F.softmax(att_score, dim=1)  # Apply softmax to get attention weights

        # Compute attention output (context vector)
        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # Weighted sum of LSTM outputs
        reps = self.tanh(reps)  # Apply Tanh activation
        return reps

    # CNN Layer: Apply convolutions with multiple kernel sizes
    def cnn_layer(self, h):
        h = h.permute(0, 2, 1)  # Change shape to fit Conv1D input (B, H, L)
        cnn_outs = [F.relu(conv(h)) for conv in self.convs]  # Apply each convolution
        # Perform global max pooling on each convolution's output
        pooled_outs = [F.max_pool1d(cnn_out, kernel_size=cnn_out.size(2)).squeeze(2) for cnn_out in cnn_outs]
        cnn_features = torch.cat(pooled_outs, dim=1)  # Concatenate CNN outputs from different kernels
        return cnn_features

    # Forward Pass: The complete forward pass for the model
    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)  # Extract word indices
        mask = data[:, 1, :].view(-1, self.max_len)   # Extract mask (padding positions)

        # Embedding Layer
        emb = self.word_embedding(token)  # Convert word indices to embeddings
        emb = self.emb_dropout(emb)  # Apply dropout to embeddings

        # BiLSTM Layer
        h = self.lstm_layer(emb, mask)  # Process through BiLSTM
        h = self.lstm_dropout(h)  # Apply dropout to LSTM output

        # Attention Representation
        att_output = self.attention_layer(h, mask)  # Compute attention representation

        # CNN Representation
        cnn_output = self.cnn_layer(h)  # Get CNN features

        # Feature Fusion: Combine Attention and CNN features
        final_rep = torch.cat([att_output, cnn_output], dim=1)  # Concatenate features from attention and CNN layers

        # Fully Connected Layer
        final_rep = self.linear_dropout(final_rep)  # Apply dropout
        logits = self.dense(final_rep)  # Classify using fully connected layer

        return logits  # Return the classification output
