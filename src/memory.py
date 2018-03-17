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

    def get(self):
        return self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx

    def set(self, k, v, a, r=None):
        return tf.group(
                self.mem_keys.assign(k),
                self.mem_vals.assign(v),
                self.mem_age.assign(a),
                (self.recent_idx.assign(r) if r is not None else tf.group()))

    def clear(self):
        return tf.variables_initializer([self.mem_keys, self.mem_vals, self.mem_age,
                                         self.recent_idx])

    def get_hint_pool_idxs(self, normalized_query):
        """Get small set of idxs to compute nearest neigbor queries on.

        This is an expensive look-up on the whole memory that is used to
        avoid more expensive later on

        Args:
            normalized_query: A Tensor of shape [None, key_dim].

        Returns:
            A Tensor of shape [None, choose_k] of indices in memory
            that are closest to the queries

        """
        # look up in large memory, no gradients
        with tf.device(self.nn_device):
            similarities = tf.matmul(tf.stop_gradient(normalized_query),
                                     self.mem_keys, transpose_b=True,
                                     name='nn_mmul')
        _, hint_pool_idxs = tf.nn.top_k(
                tf.stop_gradient(similarities), k=self.choose_k, name='nn_topk')
        return hint_pool_idxs

    def query(self, query_vec, intended_output, use_recent_idx=True):
        """Queries memory for nearest neigbor.

        Args:
            query_vec: A batch of vectors to query (embedding of input to model).
            intended_output: The values that would be correct output of the memory
            use_recent_idx: Whether to always insert at least one instance of a
                correct memory fetch

        Returns:
            A tuple (result, mask, teacher_loss)
            result: The result of the memory look up
            mask: the affinity of the query to the result
            teacher_loss: The loss for training the memory module
        """







class LSHMemory(Memory):
    """Memory employing locality sensitive hashing.

    Note: Not fully tested!!
    """

    def __init__(self, key_dim, memory_size, vocab_size,
                 choose_k=256, alpha=0.1, correct_in_top=1, age_noise=8.0,
                 var_cache_device='', nn_device='',
                 num_hashes=None, num_libraries=None):
        super(LSHMemory, self).__init__(
                key_dim, memory_size, vocab_size,
                choose_k=choose_k, alpha=alpha, correct_in_top=1,
                age_noise=age_noise, var_cache_device, nn_device=nn_device)

        self.num_libraries = num_libraries or int(self.choose_k ** 0.5)
        self.num_per_hash_slot = max(1, self.choose_k // self.num_libraries)
        self.num_hashes = (num_hashes or
                           int(np.log2(self.memory_size / self.num_per_hash_slot)))
        self.num_hashes = min(max(self.num_hashes, 1), 20)
        self.num_per_hash_slot = 2 ** self.num_hashes

        # hashing vectors
        self.hash_vecs = [
                tf.get_variable(
                    'hash_vecs%d' % i, [self.num_hashes, self.key_dim],
                    dtype=tf.float32, trainable=False,
                    initializer=tf.truncated_normal_initializer(0, 1))
                for i in range(self.num_libraries)]

        # maping representing which hash slots map to which men keys
        self.hash_slots = [
                tf.get_variable(
                    'hash_slot%d' % i, [self.num_hash_slots,
                                        self.num_per_hash_slot],
                    dtype=tf.int32, trainable=False,
                    initializer=tf.random_uniform_initializer(
                    maxval=self.memory_size, dtype=tf.int32))
                for i in range(self.num_libraries)]


