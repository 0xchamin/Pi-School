import numpy as np
import mxnet as mx
#import gluonnlp as gnlp
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn, Block
from mxnet.gluon.block import HybridBlock

# TODO(sxjscience) Add mask flag to softmax operator. Think about how to accelerate the kernel
def _masked_softmax(F, att_score, mask):
    """Ignore the masked elements when calculating the softmax

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    mask : Symbol or NDArray or None
        Shape (batch_size, query_length, memory_length)
    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        att_score = F.where(mask, att_score, -1e18 * F.ones_like(att_score))
        att_weights = F.softmax(att_score, axis=-1) * mask
    else:
        att_weights = F.softmax(att_score, axis=-1)
    return att_weights


# TODO(sxjscience) In the future, we should support setting mask/att_weights as sparse tensors
class AttentionCell(HybridBlock):
    """Abstract class for attention cells. Extend the class
     to implement your own attention method.
     One typical usage is to define your own `_compute_weight()` function to calculate the weights::

        cell = AttentionCell()
        out = cell(query, key, value, mask)

    """
    def _compute_weight(self, F, query, key, mask=None):
        """Compute attention weights based on the query and the keys

        Parameters
        ----------
        F : symbol or ndarray
        query : Symbol or NDArray
            The query vectors. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        mask : Symbol or NDArray or None
            Mask the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        att_weights : Symbol or NDArray
            For single-head attention, Shape (batch_size, query_length, memory_length)
            For multi-head attentino, Shape (batch_size, num_heads, query_length, memory_length)
        """
        raise NotImplementedError

    def _read_by_weight(self, F, att_weights, value):
        """Read from the value matrix given the attention weights.

        Parameters
        ----------
        F : symbol or ndarray
        att_weights : Symbol or NDArray
            Attention weights.
            For single-head attention,
                Shape (batch_size, query_length, memory_length).
            For multi-head attention,
                Shape (batch_size, num_heads, query_length, memory_length).
        value : Symbol or NDArray
            Value of the memory. Shape (batch_size, memory_length, total_value_dim)

        Returns
        -------
        context_vec: Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        """
        return F.batch_dot(att_weights, value)

    def __call__(self, query, key, value=None, mask=None):  # pylint: disable=arguments-differ
        """Compute the attention.

        Parameters
        ----------
        query : Symbol or NDArray
            Query vector. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        value : Symbol or NDArray or None, default None
            Value of the memory. If set to None, the value will be set as the key.
            Shape (batch_size, memory_length, value_dim)
        mask : Symbol or NDArray or None, default None
            Mask of the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        context_vec : Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        att_weights : Symbol or NDArray
            Attention weights. Shape (batch_size, query_length, memory_length)
        """
        return super(AttentionCell, self).__call__(query, key, value, mask)

    def forward(self, query, key, value=None, mask=None):  # pylint: disable=arguments-differ
        if value is None:
            value = key
        if mask is None:
            return super(AttentionCell, self).forward(query, key, value)
        else:
            return super(AttentionCell, self).forward(query, key, value, mask)


    def hybrid_forward(self, F, query, key, value, mask=None):  # pylint: disable=arguments-differ
        att_weights = self._compute_weight(F, query, key, mask)
        context_vec = self._read_by_weight(F, att_weights, value)
        return context_vec, att_weights

class PreNet(gluon.Block):
    def __init__(self, **kwargs):
        super(PreNet, self).__init__(**kwargs)
        
        with self.name_scope():
            self.fc1 = nn.Dense(256, activation='relu', flatten=False)
            self.dp1 = nn.Dropout(rate=0.5)
            self.fc2 = nn.Dense(128, activation='relu', flatten=False)
            self.dp2 = nn.Dropout(rate=0.5)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        return x
    
    
class MLPAttentionCell(AttentionCell):
    r"""Concat the query and the key and use a single-hidden-layer MLP to get the attention score.
    We provide two mode, the standard mode and the normalized mode.

    In the standard mode::

        score = v tanh(W [h_q, h_k] + b)

    In the normalized mode (Same as TensorFlow)::

        score = g v / ||v||_2 tanh(W [h_q, h_k] + b)

    This type of attention is first proposed in

    .. Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate.
       ICLR 2015

    Parameters
    ----------
    units : int
    act : Activation, default nn.Activation('tanh')
    normalized : bool, default False
        Whether to normalize the weight that maps the embedded
        hidden states to the final score. This strategy can be interpreted as a type of
        "[NIPS2016] Weight Normalization".
    dropout : float, default 0.0
        Attention dropout.
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : ParameterDict or None, default None
        See document of `Block`.
    """

    def __init__(self, units, act=nn.Activation('tanh'), normalized=False, dropout=0.0,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None):
        # Define a temporary class to implement the normalized version
        # TODO(sxjscience) Find a better solution
        class _NormalizedScoreProj(HybridBlock):
            def __init__(self, in_units, weight_initializer=None, prefix=None, params=None):
                super(_NormalizedScoreProj, self).__init__(prefix=prefix, params=params)
                self.g = self.params.get('g', shape=(1,),
                                         init=mx.init.Constant(1.0 / math.sqrt(in_units)),
                                         allow_deferred_init=True)
                self.v = self.params.get('v', shape=(1, in_units),
                                         init=weight_initializer,
                                         allow_deferred_init=True)

            def hybrid_forward(self, F, x, g, v):  # pylint: disable=arguments-differ
                v = F.broadcast_div(v, F.sqrt(F.dot(v, v, transpose_b=True)))
                weight = F.broadcast_mul(g, v)
                out = F.FullyConnected(x, weight, None, no_bias=True, num_hidden=1,
                                       flatten=False, name='fwd')
                return out

        super(MLPAttentionCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._act = act
        self._normalized = normalized
        self._dropout = dropout
        with self.name_scope():
            self._dropout_layer = nn.Dropout(dropout)
            self._query_mid_layer = nn.Dense(units=self._units, flatten=False, use_bias=True,
                                             weight_initializer=weight_initializer,
                                             bias_initializer=bias_initializer,
                                             prefix='query_')
            self._key_mid_layer = nn.Dense(units=self._units, flatten=False, use_bias=False,
                                           weight_initializer=weight_initializer,
                                           prefix='key_')
            if self._normalized:
                self._attention_score = \
                    _NormalizedScoreProj(in_units=units,
                                         weight_initializer=weight_initializer,
                                         prefix='score_')
            else:
                self._attention_score = nn.Dense(units=1, in_units=self._units,
                                                 flatten=False, use_bias=False,
                                                 weight_initializer=weight_initializer,
                                                 prefix='score_')

    def _compute_weight(self, F, query, key, mask=None):
        mapped_query = self._query_mid_layer(query)
        mapped_key = self._key_mid_layer(key)
        mid_feat = F.broadcast_add(F.expand_dims(mapped_query, axis=2),
                                   F.expand_dims(mapped_key, axis=1))
        mid_feat = self._act(mid_feat)
        att_score = self._attention_score(mid_feat).reshape(shape=(0, 0, 0))
        att_weights = self._dropout_layer(_masked_softmax(F, att_score, mask))
        return att_weights



