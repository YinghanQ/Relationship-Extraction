import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Muti_Att_BLSTM(nn.Module):
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
        self.tanh = nn.Tanh()
        self.num_heads = 10  # 新增多头注意力的头数

        # Embedding layer
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,  # 使用 BiLSTM 输出的 hidden_size 作为 embed_dim
            num_heads=self.num_heads,  
            dropout=self.lstm_dropout_value,
            batch_first=True
        )

        # Dropout layers
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        # Classification layer
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        # Initialize weights
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        """ BiLSTM encoding with masking """
        lengths = torch.sum(mask.gt(0), dim=-1)  # 计算非 PAD 的长度
        lengths = lengths.cpu()
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        
        # Reshape: (batch_size, max_len, 2, hidden_size) → (batch_size, max_len, hidden_size)
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        h = torch.sum(h, dim=2)  
        return h

    def self_attention_layer(self, h, mask):
        """ Multi-Head Self-Attention Layer """
        if mask.dim() == 3:
            mask = mask.squeeze(-1)  # 去掉最后一维，使其变为 (batch_size, seq_len)

        key_padding_mask = mask.eq(0).squeeze(-1)  # 确保是 2D 形状 (batch_size, seq_len)

        # Self-attention input: (batch_size, max_len, hidden_size)
        att_output, _ = self.self_attention(h, h, h, key_padding_mask=key_padding_mask)
        
        # Apply tanh activation
        reps = self.tanh(att_output.mean(dim=1))  # (batch_size, hidden_size)
        return reps

    def forward(self, data):
        """ Forward pass """
        token = data[:, 0, :].view(-1, self.max_len)
        mask = data[:, 1, :].view(-1, self.max_len)

        # Embedding layer
        emb = self.word_embedding(token)  # (batch_size, max_len, word_dim)
        emb = self.emb_dropout(emb)

        # LSTM layer
        h = self.lstm_layer(emb, mask)  # (batch_size, max_len, hidden_size)
        h = self.lstm_dropout(h)

        # Self-Attention layer
        reps = self.self_attention_layer(h, mask)  # (batch_size, hidden_size)
        reps = self.linear_dropout(reps)

        # Classification layer
        logits = self.dense(reps)  # (batch_size, class_num)
        return logits