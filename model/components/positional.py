from __future__ import division
import sys
sys.path.append("./..")

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, DropoutWrapper, LSTMCell, GRUCell, RNNCell, LayerNormBasicLSTMCell

def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, -1)
        return E

    return _initializer

# taken from https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

    for dim in xrange(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal

    return x

def add_timing_signal_by_lstm(x, max_height=500, lstm_num_units=128, ):
    N = tf.shape(x)[0]
    H, W = tf.shape(x)[1], tf.shape(x)[2]  # image
    C = x.shape[3].value  # channels

    pos_E = tf.get_variable(shape=[max_height, C], name="pos_E", dtype=tf.float32, initializer=embedding_initializer())
    cell = LSTMCell(num_units=lstm_num_units)

    _cond = lambda _time, _: tf.less(_time, H)
    _initial_state = tf.zeros(shape=[N, 1, lstm_num_units], dtype=tf.float32)

    def _body(_time, states):
        _row = x[:, _time, :, :]

        _pos = tf.cond(tf.less(_time, tf.constant(max_height)),
                       true_fn=lambda: tf.nn.embedding_lookup(pos_E, tf.reshape(_time, shape=(-1,))),
                       false_fn=lambda: tf.nn.embedding_lookup(pos_E,
                                                               tf.reshape(tf.constant(max_height - 1), shape=(-1,))))
        _pos = tf.tile([_pos], multiples=[N, 1, 1])

        _inp = tf.concat([_row, _pos], axis=1)
        _out, _ = tf.nn.dynamic_rnn(cell, _inp, dtype=tf.float32)

        states = tf.cond(tf.equal(_time, tf.constant(0)), true_fn=lambda: _out,
                         false_fn=lambda: tf.concat([states, _out], axis=1))

        return [tf.add(_time, 1), states]

    _, _img = tf.while_loop(cond=_cond, body=_body, loop_vars=[tf.constant(0), _initial_state],
                                 shape_invariants=[tf.constant(0).get_shape(), tf.TensorShape([None, None, lstm_num_units])])



    return _img