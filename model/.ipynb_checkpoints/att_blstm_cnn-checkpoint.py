#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Att_BLSTM_CNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num
        self.emb_dropout_value = config.emb_dropout
        self.lstm_dropout_value = config.lstm_dropout
        self.linear_dropout_value = config.linear_dropout
        self.cnn_filters = config.cnn_filters

        # Embedding Layer
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,  # 是否微调词嵌入
        )

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

        # CNN Layer (多卷积核卷积)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.hidden_size, 
                      out_channels=self.cnn_filters, 
                      kernel_size=k, 
                      padding=k // 2) for k in [2, 3, 4]
        ])

        # Attention Layer
        self.tanh = nn.Tanh()
        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))

        # Dropout Layers
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        # Fully Connected Layer (拼接后的特征用于分类)
        total_features = self.hidden_size + self.cnn_filters * len(self.convs)
        self.dense = nn.Linear(total_features, self.class_num, bias=True)

        # Weight Initialization (Xavier 初始化)
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        lengths = lengths.cpu()  # 确保 lengths 在 CPU 上，否则可能出现错误
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        # h = h.view(-1, self.max_len, 2 * self.hidden_size)  # B*L*2H
        h = torch.sum(h, dim=2)  # B*L*H，双向输出拼接
        return h

    def attention_layer(self, h, mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # B*H*1
        att_score = torch.bmm(self.tanh(h), att_weight)  # B*L*H * B*H*1 -> B*L*1

        # Masking to ignore padding positions
        mask = mask.unsqueeze(dim=-1)  # B*L*1
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # B*L*1
        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        # Attention output (context vector)
        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H
        reps = self.tanh(reps)  # B*H
        return reps

    def cnn_layer(self, h):
        # h: (B, L, H) → (B, H, L) 适配 Conv1D 输入
        h = h.permute(0, 2, 1)
        cnn_outs = [F.relu(conv(h)) for conv in self.convs]
        # 每个卷积核的输出进行全局最大池化 (B, C, L) -> (B, C)
        pooled_outs = [F.max_pool1d(cnn_out, kernel_size=cnn_out.size(2)).squeeze(2) for cnn_out in cnn_outs]
        cnn_features = torch.cat(pooled_outs, dim=1)  # (B, C1 + C2 + C3)
        return cnn_features

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)  # 词索引
        mask = data[:, 1, :].view(-1, self.max_len)   # Mask (padding 位置为 0)

        # Embedding
        emb = self.word_embedding(token)  # (B, L, D)
        emb = self.emb_dropout(emb)

        # BiLSTM
        h = self.lstm_layer(emb, mask)  # (B, L, H)
        h = self.lstm_dropout(h)

        # Attention Representation
        att_output = self.attention_layer(h, mask)  # (B, H)

        # CNN Representation
        cnn_output = self.cnn_layer(h)  # (B, C1 + C2 + C3)

        # Feature Fusion (拼接 BiLSTM + Attention + CNN)
        final_rep = torch.cat([att_output, cnn_output], dim=1)  # (B, H + CNN_Feature)

        # Fully Connected Layer
        final_rep = self.linear_dropout(final_rep)
        logits = self.dense(final_rep)  # 分类输出

        return logits
