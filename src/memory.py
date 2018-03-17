"""Memory module for storing "nearest neighbors"

Implements a key-value memory for generalized one-shot learning
as described in the paper
"Learning to remember rare events"
by Lukasz Kaiser, Ofir, Nachum, Aurko Roy, Samy Bengio,
published as a conference paper at ICLR 2017
"""

import numpy as np
import tensorflow as tf


class Memory():
    """Memory module."""

    def __init__(self, key_dim, memory_size, vocab_size,
                 choose_k=256, alpha=0.1, correct_in_top=1,
                 age_noise=8.0, var_cache_device='', nn_device=''):
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.vocab_size = vocab_size
        self.choose_k = min(choose_k, memory_size)
        self.alpha = alpha
        self.correct_in_top = correct_in_top
        self.age_noise = age_noise
        self.var_cache_device = var_cache_device # Variables are cached here.
        self.nn_device = nn_device # Device to perform nearest neighbor matmul

        caching_device = var_cache_device if var_cache_device else None
        self.update_memory = tf.constant(True) # Can be fed "fasle" if needed
        self.mem_keys = tf.get_variable(
                'memkys', [self.memory_size, self.key_dim],
                trainable=False,
                initializer=tf.random_uniform_initializer(-0.0, 0.0),
                caching_device=caching_device)
        self.mem_vals = tf.get_variable(
                'memvals', [self.memory_size], dtype=tf.int32,
                trainable=False,
                initializer=tf.constant_initializer(0, tf.int32),
                caching_device=caching_device)
        self.recent_idx = tf.get_variable(
                'recent_idx', [self.vocab_size], dtype=tf.int32,
                trainable=False,
                initializer=tf.constant_initializer(0, tf.int32))

        # variable for projecting query vecotr into memory key
        self.query_proj = tf.get_variable(
                'memory_query_proj', [self.key_dim, self.key_dim],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0, 0.01),
                caching_device=caching_device)


class LSHMemory(Memory):
    """Memory employing locality sensitive hashing.

    Note: Not fully tested!!
    """

    def __init__(self, key_dim, memory_size, vocab_size,
                 choose_k=256

