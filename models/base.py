# encoding=utf-8
from argparse import Namespace
from typing import List, Tuple

import math
import torch
import logging
from common import FLOAT_TYPE
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)


class BaseModel(nn.Module, ABC):
    def __init__(self, *args):
        super(BaseModel, self).__init__()

    @staticmethod
    @abstractmethod
    def prepare_model_params(args: Namespace) -> Namespace:
        """
        construct parameters for __init__ function from model args
        """
        pass

    @staticmethod
    def log_args(**kwargs):
        # remove unwanted keys
        unwanted_keys = ['self', '__class__']
        for key in unwanted_keys:
            if key in kwargs:
                kwargs.pop(key)

        logging.info("Create model using parameters:")
        for key, value in kwargs.items():
            logging.info("{}={}".format(key, value))

    @abstractmethod
    def init_pretrain_embeddings(self, freeze: bool):
        pass

    @classmethod
    def load(cls, model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        if isinstance(args, Namespace):
            model = cls(**vars(args))
        else:
            model = cls(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str, args: dict):
        logging.info('save model parameters to [%s]' % path)
        logging.info('model args:\n{}'.format(args))
        params = {
            'args': args,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool, batch_first: bool,
                 dropout: float):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.input_size1=input_size;
        self.hidden_size1=hidden_size;

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_lens: List[int], enforce_sorted: bool = True) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        packed_input = pack_padded_sequence(x, x_lens, batch_first=self.rnn.batch_first, enforce_sorted=enforce_sorted)
        self.rnn.flatten_parameters()

        encodings, (last_state, last_cell) = self.rnn(packed_input)
        encodings, _ = pad_packed_sequence(encodings, batch_first=self.rnn.batch_first)
        return encodings, (last_state, last_cell)

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, bias: bool = True):
        super(LSTMCell, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size=input_size,
                                    hidden_size=hidden_size,
                                    bias=bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

        self.attention=selfAttention(4,input_size,hidden_size)

    def forward(self, x: Tensor, h_tm1: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        h_t, cell_t = self.rnn_cell(x, h_tm1)

        return h_t, cell_t


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def init_bias(self, value: float):
        self.linear.bias.data.fill_(value)

    def forward(self, x: Tensor):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


def permute_lstm_output(encodings: torch.Tensor, last_state: torch.Tensor, last_cell: torch.Tensor):
    # (batch_size, sent_len, hidden_size)
    encodings = encodings.permute(1, 0, 2)
    # (batch_size, hidden_size * directions * #layers)
    last_state = torch.cat([s.squeeze(0) for s in last_state.split(1, dim=0)], dim=-1)
    last_cell = torch.cat([c.squeeze(0) for c in last_cell.split(1, dim=0)], dim=-1)
    return encodings, last_state, last_cell


def get_sent_masks(max_len: int, sent_lens: List[int], device: torch.device):
    src_sent_masks = torch.zeros(len(sent_lens), max_len, dtype=FLOAT_TYPE)
    for e_id, l in enumerate(sent_lens):
        # make all paddings to 1
        src_sent_masks[e_id, l:] = 1
    return src_sent_masks.to(device)

class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MatchAttention(nn.Module):
    def __init__(self, feature_dim_1, feature_dim_2):
        super(MatchAttention, self).__init__()


    def forward(self, features_1, features_2):
        similarity_matrix = torch.matmul(features_1, features_2.transpose(-2, -1))
        attention_weights_1 = F.softmax(similarity_matrix, dim=-1)
        attention_weights_2 = F.softmax(similarity_matrix, dim=-1)
        attended_features_1 = torch.matmul(attention_weights_1, features_2)
        attended_features_2 = torch.matmul(attention_weights_2.transpose(-2,-1), features_1)
        return attended_features_1, attended_features_2






