import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BLSTM_CNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec  # Word vector embeddings
        self.class_num = class_num  # Number of output classes

        # Hyperparameters
        self.max_len = config.max_len  # Maximum sequence length
        self.word_dim = config.word_dim  # Dimension of word embeddings
        self.hidden_size = config.hidden_size  # Hidden size for LSTM
        self.layers_num = config.layers_num  # Number of LSTM layers
        self.emb_dropout_value = config.emb_dropout  # Dropout rate for embeddings
        self.lstm_dropout_value = config.lstm_dropout  # Dropout rate for LSTM layer
        self.linear_dropout_value = config.linear_dropout  # Dropout rate for fully connected layer

        # Embedding layer to convert input words into vectors
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,  # Allow fine-tuning the word embeddings during training
        )

        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.word_dim,  # Input dimension (word embedding size)
            hidden_size=self.hidden_size,  # Output dimension (LSTM hidden state size)
            num_layers=self.layers_num,  # Number of LSTM layers
            bias=True,  # Use bias in the LSTM layers
            batch_first=True,  # Batch dimension comes first in the input tensor
            dropout=0,  # Dropout between LSTM layers (set to 0 here)
            bidirectional=True,  # Use bidirectional LSTM
        )

        # Convolutional layers to extract features from LSTM output
        self.conv1 = nn.Conv1d(self.hidden_size * 2, self.hidden_size, kernel_size=3)
        self.conv2 = nn.Conv1d(self.hidden_size * 2, self.hidden_size, kernel_size=4)
        self.conv3 = nn.Conv1d(self.hidden_size * 2, self.hidden_size, kernel_size=5)

        # Dropout layers to prevent overfitting
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)  # Dropout for word embeddings
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)  # Dropout for LSTM output
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)  # Dropout for the final output

        # Fully connected layer for classification
        self.dense = nn.Linear(
            in_features=self.hidden_size * 3,  # Input size: concatenation of 3 pooled CNN outputs
            out_features=self.class_num,  # Output size: number of classes
            bias=True  # Use bias in the fully connected layer
        )

        # Initialize weights for the dense layer using Xavier normal initialization
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)  # Initialize bias to 0

    def lstm_layer(self, x, mask):
        """ BiLSTM encoding with masking """
        lengths = torch.sum(mask.gt(0), dim=-1)  # Compute the lengths of sequences, excluding padding
        lengths = lengths.cpu()  # Move lengths to CPU for packing
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)  # Pack sequences for LSTM
        h, (_, _) = self.lstm(x)  # Forward pass through the BiLSTM layer
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)  # Unpack sequences
        return h  # Return the hidden states for all time steps

    def cnn_layer(self, h):
        """ Convolutional layers to extract features from LSTM output """
        # Permute the LSTM output to fit the Conv1d input shape: (batch_size, hidden_size * 2, max_len)
        h = h.permute(0, 2, 1)
        
        # Apply 3 different convolutional layers with ReLU activation
        conv1_out = F.relu(self.conv1(h))  # Output shape: (batch_size, hidden_size, max_len-2)
        conv2_out = F.relu(self.conv2(h))  # Output shape: (batch_size, hidden_size, max_len-3)
        conv3_out = F.relu(self.conv3(h))  # Output shape: (batch_size, hidden_size, max_len-4)

        # Apply max pooling to each convolutional output
        pool1 = F.max_pool1d(conv1_out, conv1_out.size(2)).squeeze(2)  # Pooling over length dimension
        pool2 = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, conv3_out.size(2)).squeeze(2)

        # Concatenate the pooled features from all 3 convolutions
        reps = torch.cat([pool1, pool2, pool3], dim=1)  # Output shape: (batch_size, hidden_size * 3)
        return reps

    def forward(self, data):
        """ Forward pass for the network """
        token = data[:, 0, :].view(-1, self.max_len)  # Extract tokens from input data (batch_size, max_len)
        mask = data[:, 1, :].view(-1, self.max_len)  # Extract mask (batch_size, max_len)

        # Embedding layer
        emb = self.word_embedding(token)  # Get the word embeddings for the tokens
        emb = self.emb_dropout(emb)  # Apply dropout to embeddings

        # LSTM layer
        h = self.lstm_layer(emb, mask)  # Get LSTM hidden states for each time step
        h = self.lstm_dropout(h)  # Apply dropout to LSTM output
        
        # CNN layer
        reps = self.cnn_layer(h)  # Extract features from LSTM output using CNN
        reps = self.linear_dropout(reps)  # Apply dropout before the fully connected layer

        # Classification layer
        logits = self.dense(reps)  # Compute the class logits (batch_size, class_num)
        return logits  # Return the raw class logits for each sample
