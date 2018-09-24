import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


from .utils.general import Config, Progbar, minibatches
from .utils.image import pad_batch_images
from .utils.text import pad_batch_formulas
from .evaluation.text import score_files, write_answers, truncate_end


from .encoder import Encoder
# from .decoder import Decoder
# from .encoder_inception import Encoder
from .decoder_ctc import Decoder
from .base import BaseModel


class Img2SeqCtcModel(BaseModel):
    """Specialized class for Img2Seq Model"""

    def __init__(self, config, dir_output, vocab):
        """
        Args:
            config: Config instance defining hyperparams
            vocab: Vocab instance defining useful vocab objects like tok_to_id

        """
        super(Img2SeqCtcModel, self).__init__(config, dir_output)
        self._vocab = vocab


    def build_train(self, config):
        """Builds model"""
        self.logger.info("Building model...")

        # self.seq_len = [self._config.max_length_formula] * self._config.batch_size
        self.encoder = Encoder(self._config)
        self.decoder = Decoder(self._config, self._vocab.n_tok)

        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()

        self._add_train_op(config.lr_method, self.lr, self.loss,
                config.clip)
        self.init_session()

        self.logger.info("- done.")


    def build_pred(self):
        self.logger.info("Building model...")

        self.encoder = Encoder(self._config)
        self.decoder = Decoder(self._config, self._vocab.n_tok)

        self._add_placeholders_op()
        self._add_pred_op()
        # self._add_loss_op()

        self.init_session()

        self.logger.info("- done.")



    def _add_placeholders_op(self):
        """
        Add placeholder attributes
        """
        # hyper params
        self.lr = tf.placeholder(tf.float32, shape=(),
            name='lr')
        self.dropout = tf.placeholder(tf.float32, shape=(),
            name='dropout')
        self.training = tf.placeholder(tf.bool, shape=(),
            name="training")


        # input of the graph
        self.img = tf.placeholder(tf.uint8, shape=(None, 48, None, 1),
            name='img')
        # self.formula = tf.placeholder(tf.int32, shape=(None, None),
        #     name='formula')
        self.formula = tf.sparse_placeholder(tf.int32)
        # self.formula_length = tf.placeholder(tf.int32, shape=(None, ),
        #     name='formula_length')

        # tensorboard
        tf.summary.scalar("lr", self.lr)

    def _label2sparse(self,sequence):

        indices = []
        values = []

        for index, seq in enumerate(sequence):
            indices.extend(zip([index] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int32)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(sequence), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape


    def _get_feed_dict(self, img, training, formula=None, lr=None, dropout=1):
        """Returns a dict"""
        img = pad_batch_images(img)

        fd = {
            self.img: img,
            self.dropout: dropout,
            self.training: training,
        }

        if formula is not None:
            # formula, formula_length = pad_batch_formulas(formula,
            #         self._vocab.id_pad, self._vocab.id_end)
            formula = [f if len(f) >0 else [self._vocab.id_end] for f in formula]
            fd[self.formula] = self._label2sparse(formula)
            # print(formula)
            # fd[self.formula_length] = formula_length
        if lr is not None:
            fd[self.lr] = lr

        return fd


    def _add_pred_op(self):
        """Defines self.pred"""
        encoded_img = self.encoder(self.training, self.img, self.dropout)
        train, test = self.decoder(self.training, encoded_img, self.formula,
                self.dropout)

        self.pred_train = train
        self.pred_test  = test[0]
        self.dense_decoded = test[1]
        self.seq_len = test[2]


    def _add_loss_op(self):
        """Defines self.loss"""

        loss = tf.nn.ctc_loss(labels=self.formula, inputs=self.pred_train, sequence_length=self.seq_len)
        self.loss = tf.reduce_mean(loss)

        # error rate
        acc_list = tf.edit_distance(tf.cast(self.pred_test[0], tf.int32), self.formula, normalize=True)
        acc_rate = tf.reduce_mean(acc_list)

        self.predict_accuracy = tf.maximum(1 - acc_rate, 0.0)

        tf.summary.scalar('predict_accuracy', self.predict_accuracy)
        tf.summary.scalar("loss", self.loss)



    def _run_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training

        Args:
            config: Config instance
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        # logging
        batch_size = config.batch_size
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        prog = Progbar(nbatches)
        train_set.shuffle()

        # iterate over dataset
        for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
            # get feed dict
            fd = self._get_feed_dict(img, training=True, formula=formula,
                    lr=lr_schedule.lr, dropout=config.dropout)

            # update step
            _, loss_eval, acc = self.sess.run([self.train_op, self.loss, self.predict_accuracy],
                    feed_dict=fd)
            prog.update(i + 1, [("loss", loss_eval), ("acc",
                    acc), ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

        # logging
        self.logger.info("- Training: {}".format(prog.info))

        # evaluation
        config_eval = Config({"dir_answers": self._dir_output + "formulas_val/",
                "batch_size": config.batch_size})
        scores = self.evaluate(config_eval, val_set)
        score = scores[config.metric_val]
        lr_schedule.update(score=score)

        return score


    def _run_evaluate(self, config, test_set):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """

        # iterate over the dataset
        acc_batch_total = 0
        loss_batch_total = 0
        batch_size = config.batch_size

        batch_count = 0
        nbatches = len(test_set) // batch_size
        prog = Progbar(nbatches)

        for i, (img, formula) in enumerate(minibatches(test_set, batch_size)):
            batch_count += 1
            fd = self._get_feed_dict(img, training=False, formula=formula,
                    dropout=1)
            loss_eval, acc = self.sess.run([self.loss, self.predict_accuracy],
                                           feed_dict=fd)
            prog.update(i + 1, [("loss", loss_eval), ("acc",
                                                      acc)])

            acc_batch_total = acc_batch_total + acc
            loss_batch_total = loss_batch_total + loss_eval

        # logging
        self.logger.info("- Evaluating: {}".format(prog.info))

        accuracy = acc_batch_total / batch_count
        val_loss = loss_batch_total / batch_count
        scores = {'loss':val_loss, 'acc':accuracy}

        return scores


    def predict_batch(self, images):

        fd = self._get_feed_dict(images, training=False, dropout=1)
        preds, = self.sess.run([self.dense_decoded], feed_dict=fd)

        hyps = []
        for i, pred in enumerate(preds):
            # p = truncate_end(pred, self._vocab.id_end)
            p = pred
            p = "".join([self._vocab.id_to_tok[idx] for idx in p])
            hyps.append(p)

        return hyps


    def predict(self, img):
        preds = self.predict_batch([img])
        return preds
