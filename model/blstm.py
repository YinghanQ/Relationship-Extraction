import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BLSTM(nn.Module):  # BiLSTM model without Attention
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # Hyperparameters
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num
        self.emb_dropout_value = config.emb_dropout
        self.lstm_dropout_value = config.lstm_dropout
        self.linear_dropout_value = config.linear_dropout

        # Network layers
        # Embedding layer that uses pre-trained word vectors
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,  # Allow updating the embeddings during training
        )
        
        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.word_dim,  # Word embedding dimension
            hidden_size=self.hidden_size,  # Hidden state dimension
            num_layers=self.layers_num,  # Number of LSTM layers
            bias=True,
            batch_first=True,  # Input data format is (batch_size, seq_len, input_dim)
            dropout=0,  # Dropout is handled separately in the LSTM
            bidirectional=True,  # Use bi-directional LSTM
        )
        
        # Dropout layers
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)  # Dropout for embedding layer
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)  # Dropout for LSTM output
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)  # Dropout for the final dense layer

        # Dense layer for classification
        self.dense = nn.Linear(
            in_features=self.hidden_size * 2,  # BiLSTM output will have 2 * hidden_size
            out_features=self.class_num,  # Number of output classes
            bias=True
        )

        # Initialize weights of the dense layer using Xavier normal initialization
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        """ BiLSTM encoding with masking """
        
        # Calculate the valid sequence lengths (non-PAD elements)
        lengths = torch.sum(mask.gt(0), dim=-1)  # lengths of valid tokens in each sequence
        lengths = lengths.cpu()  # Move lengths to CPU

        # Pack the padded sequences for processing by LSTM
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM output
        h, (hn, _) = self.lstm(x)  # hn shape: (num_layers * 2, batch_size, hidden_size)
        
        # Unpack the sequences back to padded form
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)

        # Extract the last valid hidden state for each sequence in the batch
        batch_size = h.shape[0]
        last_hidden_states = []
        for i in range(batch_size):
            last_idx = lengths[i] - 1  # Find the index of the last valid token
            last_hidden_states.append(h[i, last_idx, :])  # Get the hidden state for this token

        last_hidden_states = torch.stack(last_hidden_states)  # Stack to form a tensor of shape (batch_size, hidden_size * 2)
        
        return last_hidden_states

    def forward(self, data):
        """ Forward pass of the model """
        
        # Extract token data and mask from input
        token = data[:, 0, :].view(-1, self.max_len)  # Token sequence
        mask = data[:, 1, :].view(-1, self.max_len)  # Mask indicating valid positions in the sequence

        # Word embedding lookup
        emb = self.word_embedding(token)  # (batch_size, max_len, word_dim)
        emb = self.emb_dropout(emb)  # Apply dropout to embeddings

        # Process the embedding through the LSTM layer
        h = self.lstm_layer(emb, mask)  # Get the last valid hidden state for each sequence
        h = self.lstm_dropout(h)  # Apply dropout to LSTM output

        # Apply dropout before the classification layer
        reps = self.linear_dropout(h)

        # Final classification layer
        logits = self.dense(reps)  # (batch_size, class_num)
        
        return logits
