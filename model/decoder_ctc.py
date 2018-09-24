import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import GRUCell, LSTMCell


class Decoder(object):
    """Implements this paper https://arxiv.org/pdf/1609.04938.pdf"""

    def __init__(self, config, n_tok, hidden_neuron = 512, layer_num = 1):
        self._config = config
        self._n_tok = n_tok
        self._hidden_neuron = hidden_neuron
        self._layer_num = layer_num

    def __call__(self, training, img, formula, dropout):
        """Decodes an image into a sequence of token

        Args:
            training: (tf.placeholder) bool
            img: encoded image (tf.Tensor) shape = (N, H, W, C)
            formula: (tf.placeholder), shape = (N, T)

        Returns:
            pred_train: (tf.Tensor), shape = (?, ?, vocab_size) logits of each class
            pret_test: (structure)
                - pred.test.logits, same as pred_train
                - pred.test.ids, shape = (?, config.max_length_formula)

        """

        batch_size = tf.shape(img)[0]
        vocab_size = self._n_tok

        feature_h = img.shape[1]
        num_channels = img.shape[3]

        cnn_out = tf.transpose(img, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
        # print(cnn_out.shape)
        # exit()
        # print(self._seq_len)
        cnn_out = tf.reshape(cnn_out, [batch_size, -1, feature_h * num_channels])

        rnn_cells_fw = [tf.contrib.rnn.LSTMCell(self._hidden_neuron, state_is_tuple=True) for _ in range(self._layer_num)]
        rnn_cells_bw = [tf.contrib.rnn.LSTMCell(self._hidden_neuron, state_is_tuple=True) for _ in range(self._layer_num)]
        # rnn_cells_fw = [tf.contrib.rnn.GRUCell(self._hidden_neuron) for _ in
        #                 range(self._layer_num)]
        # rnn_cells_bw = [tf.contrib.rnn.GRUCell(self._hidden_neuron) for _ in
        #                 range(self._layer_num)]

        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=rnn_cells_fw,
                                                                       cells_bw=rnn_cells_bw,
                                                                       inputs=cnn_out,
                                                                       dtype=tf.float32) # [batch_size, max_stepsize, FLAGS.num_hidden]

        in_shape = tf.shape(cnn_out)
        batch_s, max_timesteps = in_shape[0], in_shape[1]
        outputs = tf.reshape(outputs, [-1, self._hidden_neuron * 2])
        seq_len = tf.fill((batch_size,), max_timesteps) #[max_timesteps] * batch_size

        w = tf.get_variable(name='w',
                            shape=[self._hidden_neuron * 2, vocab_size],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=[vocab_size],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())

        logits = tf.matmul(outputs, w) + b

        logits = tf.reshape(logits, [batch_s, -1, vocab_size])
        logits = tf.transpose(logits, [1, 0, 2])
        train_outputs = logits

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, sequence_length=seq_len, merge_repeated=False, beam_width=20)
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        return train_outputs, (decoded, dense_decoded, seq_len)

