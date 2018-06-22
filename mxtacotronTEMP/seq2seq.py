from __future__ import unicode_literals

import argparse
import random
import unicodedata
import string
import re

import numpy as np
import mxnet as mx
from io import open
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F





class AttnDecoderRNN(Block):
    def __init__(self, hidden_size, output_size, n_layers, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        with self.name_scope():
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
            self.attn = nn.Dense(self.max_length, in_units=self.hidden_size * 2)
            self.attn_combine = nn.Dense(self.hidden_size, in_units=self.hidden_size * 2)
            if self.dropout_p > 0:
                self.dropout = nn.Dropout(self.dropout_p)
            self.gru = rnn.GRU(self.hidden_size, input_size=self.hidden_size)
            self.out = nn.Dense(self.output_size, in_units=self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        #input shape, (1,)
        embedded = self.embedding(input)
        if self.dropout_p > 0:
            embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(F.concat(embedded, hidden[0].flatten(), dim=1)))
        attn_applied = F.batch_dot(attn_weights.expand_dims(0),
                                 encoder_outputs.expand_dims(0))

        output = F.concat(embedded.flatten(), attn_applied.flatten(), dim=1)
        output = self.attn_combine(output).expand_dims(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = self.out(output)

        return output, hidden, attn_weights

    def initHidden(self, ctx):
        return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]


class EncoderRNN(Block):
    def __init__(self, input_size, hidden_size, n_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        with self.name_scope():
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = rnn.GRU(hidden_size, input_size=self.hidden_size)

    def forward(self, input, hidden):
        ##input shape, (seq,)
        output = self.embedding(input).swapaxes(0, 1)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, ctx):
        return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]


