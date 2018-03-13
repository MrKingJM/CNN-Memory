r"""Scipt for training model

Simple command to get up and running:
    python3 train.py

"""

import logging
import os

import tensorflow as tf

import data_utils

FALGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('rep_dim', 128,
                        'dimension of keys to use in memory')
tf.flags.DEFINE_integer('episode_length', 100, 'length of episode')
tf.flags.DEFINE_integer('episode_width', 5,
                        'number of distinct labels in a single episode')
tf.flags.DEFINE_integer('memory_size', None, 'number of slots in memory.'


class Trainer():
    """Class than takes care of training, validating, and checkpointing model."""

    def __init__(self, train_data, valid_data, input_dim, output_dim=None):
        self.train_data = train_data
        self.valid_data = valid_data
        self.input_dim = input_dim

        self.rep_dim = FLAGS.rep_dim

def main(unused_argv):
    train_data, valid_data = data_utils.get_data()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
