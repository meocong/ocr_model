import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.contrib.slim.nets import inception, resnet_v2

from .components.positional import add_timing_signal_nd


class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config):
        self._config = config


    def __call__(self, training, img, dropout):
        """Applies convolutions to the image

        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type
                tf.uint8

        Returns:
            the encoded images, shape = (?, h', w', c')

        """
        img = tf.cast(img, tf.float32) / 255.

        # out, _ = inception.inception_v2_base(img, final_endpoint='MaxPool_3a_3x3',
        #                                   scope="convolutional_encoder")
        out, _ = inception.inception_v1_base(img, final_endpoint='MaxPool_3a_3x3',
                                          scope="convolutional_encoder")

        if self._config.positional_embeddings:
            # from tensor2tensor lib - positional embeddings
            out = add_timing_signal_nd(out)

        return out
