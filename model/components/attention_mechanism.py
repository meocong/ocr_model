import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, DropoutWrapper, LSTMCell, GRUCell, RNNCell
#from ..decoder import embedding_initializer
#from ..decoder import embedding_initializer

def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, -1)
        return E

    return _initializer

class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1):
        """Stores the image under the right shape.

        We loose the H, W dimensions and merge them into a single
        dimension that corresponds to "regions" of the image.

        Args:
            img: (tf.Tensor) image
            dim_e: (int) dimension of the intermediary vector used to
                compute attention
            tiles: (int) default 1, input to context h may have size
                    (tile * batch_size, ...)

        """
        # dimensions


        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N    = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2] # image
            C    = img.shape[3].value                 # channels

            max_height = 500
            pos_E = tf.get_variable(shape=[500, C], name="pos_E", dtype=tf.float32, initializer=embedding_initializer())
            cell  = LSTMCell(num_units=dim_e, _scope="lstm_pos_E_cell")

            _cond = lambda _time, _: tf.less(_time, H)
            _initial_state = tf.zeros(shape=[N,1,dim_e], dtype=tf.float32)

            def _body(_time, states):
                _row = img[:, _time, :, :]

                _pos = tf.cond(tf.less(_time, tf.constant(max_height)),
                               true_fn=lambda : tf.nn.embedding_lookup(pos_E, tf.reshape(_time, shape=(-1,))),
                               false_fn= lambda: tf.nn.embedding_lookup(pos_E, tf.reshape(tf.constant(max_height-1),shape=(-1,))) )
                _pos = tf.tile([_pos], multiples=[N, 1, 1]) #tf.tile([_pos], multiples=[N, tf.shape(_row)[1], 1])

                _inp = tf.concat([_row, _pos], axis=1)
                _out,_ = tf.nn.dynamic_rnn(cell, _inp, dtype=tf.float32)

                states = tf.cond(tf.equal(_time,tf.constant(0)), true_fn= lambda : _out,
                                 false_fn= lambda : tf.concat([states, _out], axis=1))

                return [tf.add(_time,1), states]

            _, self._img = tf.while_loop(cond=_cond, body=_body, loop_vars=[tf.constant(0), _initial_state],
                                        shape_invariants=[tf.constant(0).get_shape(), tf.TensorShape([None, None, dim_e])])


            #self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        self._n_regions = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        self._dim_e = dim_e
        self._tiles = tiles
        self._scope_name = "att_mechanism"

        self._att_img = self._img
        # attention vector over the image
        # self._att_img = tf.layers.dense(
        #     inputs=self._img,
        #     units=self._dim_e,
        #     use_bias=False,
        #     name="att_img")


    def context(self, h):
        """Computes attention

        Args:
            h: (batch_size, num_units) hidden state

        Returns:
            c: (batch_size, channels) context vector

        """
        with tf.variable_scope(self._scope_name):
            if self._tiles > 1:
                att_img = tf.expand_dims(self._att_img, axis=1)
                att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
                att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
                        self._dim_e])
                img = tf.expand_dims(self._img, axis=1)
                img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
                img = tf.reshape(img, shape=[-1, self._n_regions,
                        self._n_channels])
            else:
                att_img = self._att_img
                img     = self._img

            # computes attention over the hidden vector
            att_h = tf.layers.dense(inputs=h, units=self._dim_e, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1],
                    dtype=tf.float32)
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            e = tf.matmul(att_flat, att_beta)
            e = tf.reshape(e, shape=[-1, self._n_regions])

            # compute weights
            a = tf.nn.softmax(e)
            a = tf.expand_dims(a, axis=-1)
            c = tf.reduce_sum(a * img, axis=1)

            return c


    def initial_cell_state(self, cells):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        if type(cells) in [MultiRNNCell, DropoutWrapper]:
            initial_state_cell = []

            lst_cells = cells._cells if type(cells) is MultiRNNCell else cells._cell._cells

            for _, cell in enumerate(lst_cells):
                _states_0 = []

                with tf.variable_scope("__init_hidden_state_attn_%d" % _):
                    for hidden_name in cell._state_size._fields:
                        hidden_dim = getattr(cell._state_size, hidden_name)
                        h = self.initial_state(hidden_name, hidden_dim)
                        _states_0.append(h)

                    initial_state_cell += [type(cell.state_size)(*_states_0)]

            return tuple(initial_state_cell)

        elif type(cells) in [LSTMCell, GRUCell, RNNCell]:
            _states_0 = []

            for hidden_name in cells._state_size._fields:
                hidden_dim = getattr(cells._state_size, hidden_name)
                h = self.initial_state(hidden_name, hidden_dim)
                _states_0.append(h)

            initial_state_cell = type(cells.state_size)(*_states_0)

            return initial_state_cell
        else:
            raise Exception("initial_cell_state unknown cell type")


    def initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels,
                    dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h