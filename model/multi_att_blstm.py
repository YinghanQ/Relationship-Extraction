import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Multi_Att_BLSTM(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # Hyperparameters
        self.max_len = config.max_len  # Maximum sequence length
        self.word_dim = config.word_dim  # Word embedding dimension
        self.hidden_size = config.hidden_size  # Hidden size for LSTM
        self.layers_num = config.layers_num  # Number of LSTM layers
        self.emb_dropout_value = config.emb_dropout  # Dropout rate for embedding layer
        self.lstm_dropout_value = config.lstm_dropout  # Dropout rate for LSTM layer
        self.linear_dropout_value = config.linear_dropout  # Dropout rate for linear layer
        self.tanh = nn.Tanh()  # Tanh activation function
        self.num_heads = 10  # Number of attention heads in multi-head attention

        # Embedding layer
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,  # Initialize with pre-trained word vectors
            freeze=False,  # Allow training the word embeddings
        )

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.word_dim,  # Input size from the word embeddings
            hidden_size=self.hidden_size,  # LSTM hidden size
            num_layers=self.layers_num,  # Number of LSTM layers
            bias=True,  # Bias term
            batch_first=True,  # Batch size as the first dimension
            dropout=0,  # No dropout between LSTM layers
            bidirectional=True,  # Bidirectional LSTM
        )

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,  # Use BiLSTM output hidden size as embedding dimension
            num_heads=self.num_heads,  # Number of attention heads
            dropout=self.lstm_dropout_value,  # Dropout rate for attention
            batch_first=True  # Input format: (batch_size, seq_len, embed_dim)
        )

        # Dropout layers
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)  # Dropout for word embeddings
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)  # Dropout for LSTM output
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)  # Dropout for final representations

        # Classification layer
        self.dense = nn.Linear(
            in_features=self.hidden_size,  # Hidden size from LSTM
            out_features=self.class_num,  # Number of classes for classification
            bias=True
        )

        # Initialize weights using Xavier normal initialization for dense layer
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        """ BiLSTM encoding with masking """
        # Compute the lengths of non-padding elements
        lengths = torch.sum(mask.gt(0), dim=-1)  # Count non-padding tokens in each sequence
        lengths = lengths.cpu()  # Move lengths to CPU for processing
        
        # Pack the padded sequence for LSTM
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Get the output of the LSTM, discard hidden state (hn) and cell state (cn)
        h, (_, _) = self.lstm(x)
        
        # Pad the packed sequence to restore the original shape
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        
        # Reshape the hidden states and sum the forward and backward LSTM outputs
        h = h.view(-1, self.max_len, 2, self.hidden_size)  # Shape: (batch_size, seq_len, 2, hidden_size)
        h = torch.sum(h, dim=2)  # Sum across the bidirectional dimension
        
        return h

    def self_attention_layer(self, h, mask):
        """ Multi-Head Self-Attention Layer """
        if mask.dim() == 3:
            mask = mask.squeeze(-1)  # Remove the last dimension to make it (batch_size, seq_len)

        # Create key_padding_mask to ignore padding tokens in attention
        key_padding_mask = mask.eq(0).squeeze(-1)  # Ensure mask is 2D (batch_size, seq_len)

        # Apply multi-head attention on the LSTM output
        att_output, _ = self.self_attention(h, h, h, key_padding_mask=key_padding_mask)
        
        # Apply tanh activation and average over sequence length dimension
        reps = self.tanh(att_output.mean(dim=1))  # (batch_size, hidden_size)
        return reps

    def forward(self, data):
        """ Forward pass """
        token = data[:, 0, :].view(-1, self.max_len)  # Extract token indices
        mask = data[:, 1, :].view(-1, self.max_len)  # Extract mask for padding positions

        # Embedding layer
        emb = self.word_embedding(token)  # (batch_size, max_len, word_dim)
        emb = self.emb_dropout(emb)  # Apply embedding dropout

        # LSTM layer
        h = self.lstm_layer(emb, mask)  # (batch_size, max_len, hidden_size)
        h = self.lstm_dropout(h)  # Apply LSTM dropout

        # Self-Attention layer
        reps = self.self_attention_layer(h, mask)  # (batch_size, hidden_size)
        reps = self.linear_dropout(reps)  # Apply linear dropout

        # Classification layer
        logits = self.dense(reps)  # (batch_size, class_num)
        return logits
