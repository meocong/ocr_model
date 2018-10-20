import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, DropoutWrapper, LSTMCell, GRUCell, RNNCell
from .positional import add_timing_signal_nd, add_timing_signal_by_lstm

class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1, use_positional_embedding = True, use_positional_embedding_lstm=False,
                 n_stacked = 1):
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
            if not use_positional_embedding:
                self._img = tf.reshape(img, shape=[tf.shape(img)[0], tf.shape(img)[1] * tf.shape(img)[2], img.shape[3].value])
            else:
                if not use_positional_embedding_lstm:
                    self._img = add_timing_signal_nd(img)

                    self._img = tf.reshape(self._img, shape=[tf.shape(self._img)[0], tf.shape(self._img)[1] * tf.shape(self._img)[2],
                                                             self._img.shape[3].value])
                    self._img = tf.layers.dense(
                        inputs=self._img,
                        units=dim_e,
                        use_bias=False,
                        name="att_img")
                else:
                    self._img = add_timing_signal_by_lstm(img,lstm_num_units=dim_e)

        else:
            print("Image shape not supported")
            raise NotImplementedError

        self._fn_states = None
        if n_stacked > 0:
            with tf.variable_scope("encoder_lstm"):
                cells = [LSTMCell(num_units=dim_e) for _ in range(n_stacked)]
                cell = MultiRNNCell(cells)

                self._img, self._fn_states = tf.nn.dynamic_rnn(cell, self._img, dtype=tf.float32)
                self._att_img = self._img
        else:
            self._att_img = self._img

        self._n_regions = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        self._dim_e = dim_e
        self._tiles = tiles
        self._scope_name = "att_mechanism"
        self._use_positional_embedding = use_positional_embedding
        self._use_positional_embedding_lstm = use_positional_embedding_lstm

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

        return self._fn_states

        #
        # if type(cells) in [MultiRNNCell, DropoutWrapper]:
        #     initial_state_cell = []
        #
        #     lst_cells = cells._cells if type(cells) is MultiRNNCell else cells._cell._cells
        #
        #     for _, cell in enumerate(lst_cells):
        #         _states_0 = []
        #
        #         with tf.variable_scope("__init_hidden_state_attn_%d" % _):
        #             for hidden_name in cell._state_size._fields:
        #                 hidden_dim = getattr(cell._state_size, hidden_name)
        #                 h = self.initial_state(hidden_name, hidden_dim)
        #                 _states_0.append(h)
        #
        #             initial_state_cell += [type(cell.state_size)(*_states_0)]
        #
        #     return tuple(initial_state_cell)
        #
        # elif type(cells) in [LSTMCell, GRUCell, RNNCell]:
        #     _states_0 = []
        #
        #     for hidden_name in cells._state_size._fields:
        #         hidden_dim = getattr(cells._state_size, hidden_name)
        #         h = self.initial_state(hidden_name, hidden_dim)
        #         _states_0.append(h)
        #
        #     initial_state_cell = type(cells.state_size)(*_states_0)
        #
        #     return initial_state_cell
        #
        # else:
        #     raise Exception("initial_cell_state unknown cell type")


    def initial_state(self, name, dim, reuse=False):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name, reuse=reuse):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels,dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h