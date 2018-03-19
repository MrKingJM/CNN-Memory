"""Model using memory component.

The model embeds images using a standard CNN architecture.
These embeddins are used as keys to the memory component,
which returns nearest neighbors.
"""

import tensorflow as tf

import memory

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

    def __init__(self, image_size, num_channels, hidden_dim):
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.matrix_init = tf.truncated_normal_initializer(stddev=0.1)
        self.vector_init = tf.constant_initializer(0.0)

    def core_builder(self, x):
        """Embeds x using standard CNN architecture.

        Args:
            x: Batch of images as 2-d Tensor [batch_size, -1].

        Returns:
            A 2-d Tensor [batch_size, hidden_dim] of embedded images.
        """

        ch1 = 32 * 2 # number of channels in 1st layer ??
        ch2 = 64 * 2 # number of channels in 2nd layer ??

        #y kernel size is 3*3, conv1 use ch1, conv2 use ch2
        # for each kernel we use the same amount biases
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

        conv2a_weights = tf.get_variable('conv2a_w', [3, 3, ch1, ch2],
                                         initializer=self.matrix_init)
        conv2a_biases = tf.get_variable('conv2a_b', [ch2],
                                        initializer=self.vector_init)
        conv2b_weights = tf.get_variable('conv2b_w', [3, 3, ch2, ch2],
                                         initializer=self.matrix_init)
        conv2b_biases = tf.get_variable('conv2b_b', [ch2],
                                        initializer=self.vector_init)

        # fully conncted
        fc1_weights = tf.get_variable(
                'fc1_w', [self.image_size // 4 * self.image_size // 4 * ch2,
                          self.hidden_dim], initializer=self.matrix_init)
        fc1_biases = tf.get_variable('fc1_b', [self.hidden_dim],
                                     initializer=self.vector_init)

        # define model
        x = tf.reshape(x,
                       [-1, self.image_size, self.image_size, self.num_channels])
        batch_size = tf.shape(x)[0]

        conv1a = tf.nn.conv2d(x, conv1a_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1a = tf.nn.relu(tf.nn.bias_add(conv1a, conv1a_biases))
        conv1b = tf.nn.conv2d(relu1a, conv1b_weights,
                              strides=[1, 1, 1, 1], padding='SAME')
        relu1b = tf.nn.relu(tf.nn.bias_add(conv1b, conv1b_biases))

        # subsample 2*2, ksize for window, strides for walking
        pool1 = tf.nn.max_pool(relu1b, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

        conv2a = tf.nn.conv2d(pool1, conv2a_weights,
                              strides=[1, 1, 1, 1], padding='SAME')
        relu2a = tf.nn.relu(tf.nn.bias_add(conv2a, conv2a_biases))
        conv2b = tf.nn.conv2d(relu2a, conv2b_weights,
                            strides=[1, 1, 1, 1], padding='SAME')
        relu2b = tf.nn.relu(tf.nn.bias_add(conv2b, conv2b_biases))

        pool2 = tf.nn.max_pool(relu2b, ksize=[1, 2, 2, 1],
                               strides=[1,2,2,1], padding='SAME')

        reshape = tf.reshape(pool2, [batch_size, -1])
        hidden = tf.matmul(reshape, fc1_weights) + fc1_biases

        return hidden


class Model():
    """MOdel for coordinating bwtween CNN embedder and Memory module."""

    def __init__(self, input_dim, output_dim, rep_dim, memory_size, vocab_size,
                 learning_rate=0.0001, use_lsh=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rep_dim = rep_dim
        self.memory_size = memory_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.use_lsh = use_lsh

        self.embedder = self.get_embedder()
        self.memory = self.get_memory()
        self.classifier = self.get_classifier()

        self.global_step = tf.train.get_or_create_global_step()

    def train(self, x, y):
        loss, _ = self.core_builder(x, y, keep_prob=0.3)
        gradient_ops = self.training_ops(loss)
        return loss, gradient_ops

    def eval(self, x, y):
        _, y_preds = self.core_builder(x, y, keep_prob=1.0,
                                       use_recent_idx=False)
        return y_preds

    def get_xy_placeholders(self):
        return (tf.placeholder(tf.float32, [None, self.input_dim]),
                tf.placeholder(tf.int32, [None]))

    def setup(self):
        """Sets up all components of the computation graph."""

        self.x, self.y = self.get_xy_placeholders()

        with tf.variable_scope('core', reuse=None):
            self.loss, self.gradient_ops = self.train(self.x, self.y)
        with tf.variable_scope('core', reuse=True):
            self.y_preds = self.eval(self.x, self.y) # why self.y

        # setup memory "reset" ops
        (self.mem_keys, self.mem_vals,
         self.mem_age, self.recent_idx) = self.memory.get()
        self.mem_keys_reset = tf.placeholder(self.mem_keys.dtype,
                                             tf.identity(self.mem_keys).shape)
        self.mem_vals_reset = tf.placeholder(self.mem_vals.dtype,
                                             tf.identity(self.mem_vals).shape)
        self.mem_age_reset = tf.placeholder(self.mem_age.dtype,
                                            tf.identity(self.mem_age).shape)
        self.recent_idx_reset = tf.placeholder(self.recent_idx.dtype,
                                               tf.identity(self.recent_idx).shape)
        self.mem_reset_op = self.memory.set(self.mem_keys_reset,
                                            self.mem_vals_reset,
                                            self.mem_age_reset,
                                            None)

    def training_ops(self, loss):
        opt = self.get_optimizer()
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return opt.apply_gradients(zip(clipped_gradients, params),
                                   global_step=self.global_step)

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                      epsilon=1e-4)

    def get_embedder(self):
        return LeNet(int(self.input_dim ** 0.5), 1, self.rep_dim)

    def get_memory(self):
        cls = memory.LSHMemory if self.use_lsh else memory.Memory
        return cls(self.rep_dim, self.memory_size, self.vocab_size)

    def get_classifier(self):
        return BasicClassfier(self.output_dim)

    def core_builder(self, x, y, keep_prob, use_recent_idx=True):
        embeddings = self.embedder.core_builder(x)
        if keep_prob < 1.0:
            embeddings = tf.nn.dropout(embeddings, keep_prob)
        memory_val, _, teacher_loss = self.memory.query(
                embeddings, y, use_recent_idx=use_recent_idx)
        loss, y_preds = self.classifier.core_builder(memory_val, x, y)

        return loss + teacher_loss, y_preds

    def episode_step(self, sess, x, y, clear_memory=False):
        """Performs training steps on episodic input.

        Args:
            sess: A Tensorflow Session.
            x: A list of batches of images defining the episode.
            y: A list of batches of labels corresponding to x.
            clear_memory: Whether to clear the memory before episode.

        Returns:
            List of losses the same length as the episode.
        """

        outputs = [self.loss, self.gradient_ops]

        if clear_memory:
            self.clear_memory(sess)

        losses = []
        for xx, yy in zip(x, y):
            out = sess.run(outputs, feed_dict={self.x: xx, self.y:yy})
            loss = out[0]
            losses.append(loss)

        return losses
#
    def episode_predict(self, sess, x, y, clear_memory=False):
        """Predict the labels on an episode of examples.

        Args:
            sess: A tensorflow Session
            x: A list of batches of images
            y: A list of labels for the images in x
                This allows for updating the memory
            clear_memory" Whether to clear the memory before the episode

        Returns:
            List of predicted y.
        """

        cur_memory = sess.run([self.mem_keys, self.mem_vals,
                               self.mem_age])

        if clear_memory:
            self.clear_memory(sess)

        outputs = [self.y_preds]
        y_preds = []
        for xx, yy in zip(x, y):
            out = sess.run(outputs, feed_dict={self.x: xx, self.y: yy})
            y_pred = out[0]
            y_preds.append(y_pred)

        sess.run([self.mem_reset_op],
                  feed_dict={self.mem_keys_reset: cur_memory[0],
                             self.mem_vals_reset: cur_memory[1],
                             self.mem_age_reset: cur_memory[2]})

        return y_preds

    def clear_memory(self, sess):
        sess.run([self.memory.clear()])
