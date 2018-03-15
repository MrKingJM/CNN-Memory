"""Model using memory component.

The model embeds images using a standard CNN architecture.
These embeddins are used as keys to the memory component,
which returns nearest neighbors.
"""

import tensorflow as tf

#import memory

FLAG = tf.flags.FLAGS


class BasicClassfier():

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def core_builder(self, memory_val, x, y):
        del x, y
        y_pred = memory_val
        loss = 0.0

        return loss, y_pred


class LeNet():
    """Standard CNN architecture."""

    def __int__(self, image_size, num_channels, hidden_dim):
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.matrix_init = tf.truncated_normal_initializer(stddev=0.1)
        self.vactor_init = tf.constant_initializer(0.0)

    def core_builder(self, x):
        """Embeds x using standard CNN architecture.

        Args:
            x: Batch of images as 2-d Tensor [batch_size, -1].

        Returns:
            A 2-d Tensor [batch_size, hidden_dim] of embedded images.
        """

        ch1 = 32 * 2 # number of channels in 1st layer
        ch2 = 64 * 2 # number of channels in 2nd layer
        conv1a_weights = tf.get_variable('conv1a_w',
                                        [3, 3, self.num_channels, ch1],
                                        initializer=self.matrix_init)
        conv1a_biases = tf.get_variable('conv1a_b', [ch1],
                                       initializer=self.vector_init)
        conv1b_weights = tf.get_variable('conv1b_w',
                                        [3, 3, ch1, ch1],
                                        initializer=self.matrix_init)
        conv1b_biases = tf.get_variable('conv1b_b', [ch1],
                                        initializer=self.vector_init)

        conv2a_weights = tf.get_variable('conv2b_w', [3, 3, ch1, ch2],
                                         initializer=self.matrix_init)
        conv2a_biases = tf.get_variable('conv2b_b', [ch2],
                                        initializer=self.vector_init)
        conv2b_weights = tf.get_variable('conv2b_w', [3, 3, ch2, ch2],
                                         initializer=self.matrix_init)
        conv2b_biases = tf.get_variable('conv2b_b', [ch2],
                                        initializer=self.vector_init)

        # fully conncted
        fc1_weights = tf.get_variable(
                'fc1_w', [

