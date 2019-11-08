from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import tensorflow as tf
import pickle
import random
import json

data_path = '/home/amax/Desktop'


class DataSet:

    def __init__(self):
        self.IMAGE_SIZE=32
        self.NUM_EXAMPLES_FOR_TRAIN=40000
        return

    def inputs(self):
        print("======Loading data======")
        train_files = ['data_batch_%d' % d for d in range(1, 6)]
        train_data, train_label = self._load(train_files)
        train_data, train_label, valid_data, valid_label = self._split(train_data, train_label)
        test_data, test_label = self._load(['test_batch'])
        train_data = self._process(train_data)
        print("======Data Process Done======")
        return train_data, train_label, valid_data, valid_label, test_data, test_label

    def _load(self, files):
        data_dir = os.path.join(data_path, 'cifar-10-batches-py')
        with open(os.path.join(data_dir, files[0]), 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        data = batch[b'data']
        label = np.array(batch[b'labels'])
        for f in files[1:]:
            with open(os.path.join(data_dir, f), 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
            data = np.append(data, batch[b'data'], axis=0)
            label = np.append(label, batch[b'labels'], axis=0)
        # label = np.array([[float(i == label) for i in range(self.NUM_CLASSES)] for label in label])
        data = data.reshape([-1, 3, self.IMAGE_SIZE, self.IMAGE_SIZE])
        data = data.transpose([0, 2, 3, 1])
        # preprocess
        data = self._normalize(data)
        # shuffle
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data = data[index]
        label = label[index]

        return data, label

    def _split(self, data, label):
        return data[:self.NUM_EXAMPLES_FOR_TRAIN], label[:self.NUM_EXAMPLES_FOR_TRAIN], \
               data[self.NUM_EXAMPLES_FOR_TRAIN:], label[self.NUM_EXAMPLES_FOR_TRAIN:]

    def _normalize(self, x_train):
        x_train = x_train.astype('float32')
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - np.mean(x_train[:, :, :, i])) / np.std(x_train[:, :, :, i])
        return x_train

    def _process(self, x):
        x = self._random_crop(x, [32, 32], 8)
        x = self._random_flip_leftright(x)
        # x = self._cutout(x)
        return x

    def _random_crop(self, batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        if padding:
            oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                           nw:nw + crop_shape[1]]
        return np.array(new_batch)

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def _cutout(self, x):
        for i in range(len(x)):
            cut_size = random.randint(0, self.IMAGE_SIZE // 2)
            s = random.randint(0, self.IMAGE_SIZE - cut_size)
            x[i, s:s + cut_size, s:s + cut_size, :] = 0
        return x


class Evaluator:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        config=json.load(os.path.join(os.getcwd(),'nas_config.json'))
        # Global constants describing the CIFAR-10 data set.
        self.IMAGE_SIZE = 32
        self.NUM_CLASSES = 10
        self.NUM_EXAMPLES_FOR_TRAIN = 40000
        self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
        # Constants describing the training process.
        self.INITIAL_LEARNING_RATE = config['INITIAL_LEARNING_RATE']  # Initial learning rate.
        self.NUM_EPOCHS_PER_DECAY = config['NUM_EPOCHS_PER_DECAY']  # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = config['LEARNING_RATE_DECAY_FACTOR']  # Learning rate decay factor.
        self.MOVING_AVERAGE_DECAY = config['MOVING_AVERAGE_DECAY']
        self.batch_size = config['batch_size']
        self.epoch = config['epoch']
        self.weight_decay = config['weight_decay']
        self.momentum_rate = config['momentum_rate']
        self.model_path = config['model_path']
        self.train_num = 0
        self.network_num = 0
        self.max_steps = 0
        self.blocks = 0
        self.train_data, self.train_label, self.valid_data, self.valid_label, \
        self.test_data, self.test_data = DataSet().inputs()

    def _batch_norm(self, input, train_flag):
        # return input
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            updates_collections=None, is_training=train_flag)

    def _makeconv(self, inputs, hplist, node, train_flag):
        """Generates a convolutional layer according to information in hplist

        Args:
        inputs: inputing data.
        hplist: hyperparameters for building this layer
        node: number of this cell
        Returns:
        tensor.
        """
        # print('Evaluater:right now we are making conv layer, its node is %d, and the inputs is'%node,inputs,'and the node before it is ',cellist[node-1])
        with tf.variable_scope('conv' + str(node) + 'block' + str(self.blocks)) as scope:
            inputdim = inputs.shape[3]
            assert type(hplist[2]) == type(1), 'Wrong type of filter size: %s.' % str(type(hplist[2]))
            kernel = tf.get_variable('weights', shape=[hplist[2], hplist[2], inputdim, hplist[1]],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', hplist[1], initializer=tf.constant_initializer(0.0))
            bias = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            if hplist[3] == 'relu':
                conv1 = tf.nn.relu(bias, name=scope.name)
            elif hplist[3] == 'tanh':
                conv1 = tf.tanh(bias, name=scope.name)
            elif hplist[3] == 'sigmoid':
                conv1 = tf.sigmoid(bias, name=scope.name)
            elif hplist[3] == 'identity':
                conv1 = tf.identity(bias, name=scope.name)
            elif hplist[3] == 'leakyrelu':
                conv1 = tf.nn.leaky_relu(bias, name=scope.name)
            else:
                print('Wrong! %s is not a legal activation function!' % hplist[3])
        return conv1

    def _makepool(self, inputs, hplist):
        """Generates a pooling layer according to information in hplist

        Args:
            inputs: inputing data.
            hplist: hyperparameters for building this layer
        Returns:
            tensor.
        """
        if hplist[1] == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                                  strides=[1, hplist[2], hplist[2], 1], padding='SAME')
        elif hplist[1] == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                                  strides=[1, hplist[2], hplist[2], 1], padding='SAME')
        elif hplist[1] == 'global':
            return tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    def _makedense(self, inputs, hplist, train_flag):
        """Generates dense layers according to information in hplist

        Args:
                   inputs: inputing data.
                   hplist: hyperparameters for building layers
                   node: number of this cell
        Returns:
                   tensor.
        """
        i = 0
        inputs = tf.reshape(inputs, [self.batch_size, -1])

        for neural_num in hplist[1]:
            with tf.variable_scope('dense' + str(i)) as scope:
                weights = tf.get_variable('weights', shape=[inputs.shape[-1], neural_num],
                                          initializer=tf.contrib.keras.initializers.he_normal())
                # weight = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
                # tf.add_to_collection('losses', weight)
                biases = tf.get_variable('biases', [neural_num], initializer=tf.constant_initializer(0.0))
                if hplist[2] == 'relu':
                    local3 = tf.nn.relu(self._batch_norm(tf.matmul(inputs, weights) + biases, train_flag),
                                        name=scope.name)
                elif hplist[2] == 'tanh':
                    local3 = tf.tanh(tf.matmul(inputs, weights) + biases, name=scope.name)
                elif hplist[2] == 'sigmoid':
                    local3 = tf.sigmoid(tf.matmul(inputs, weights) + biases, name=scope.name)
                elif hplist[2] == 'identity':
                    local3 = tf.identity(tf.matmul(inputs, weights) + biases, name=scope.name)
            inputs = local3
            i += 1
        return inputs

    def _inference(self, images, graph_part, cellist, train_flag):  # ,regularizer):
        '''Method for recovering the network model provided by graph_part and cellist.
        Args:
          images: Images returned from Dataset() or inputs().
          graph_part: The topology structure of th network given by adjacency table
          cellist:

        Returns:
          Logits.'''
        # print('Evaluater:starting to reconstruct the network')
        nodelen = len(graph_part)
        inputs = [0 for i in range(nodelen)]  # input list for every cell in network
        inputs[0] = images
        getinput = [False for i in range(nodelen)]  # bool list for whether this cell has already got input or not
        getinput[0] = True

        for node in range(nodelen):
            # print('Evaluater:right now we are processing node %d'%node,', ',cellist[node])
            if cellist[node][0] == 'conv':
                layer = self._makeconv(inputs[node], cellist[node], node, train_flag)
                # layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            elif cellist[node][0] == 'pooling':
                layer = self._makepool(inputs[node], cellist[node])
            elif cellist[node][0] == 'dense':
                layer = self._makedense(inputs[node], cellist[node], train_flag)
            else:
                print('WRONG!!!!! Notice that you got a layer type we cant process!', cellist[node][0])
                layer = []

            # update inputs information of the cells below this cell
            for j in graph_part[node]:
                if getinput[j]:  # if this cell already got inputs from other cells precedes it
                    # padding
                    a = int(layer.shape[1])
                    b = int(inputs[j].shape[1])
                    pad = abs(a - b)
                    if layer.shape[1] > inputs[j].shape[1]:
                        tmp = tf.pad(inputs[j], [[0, 0], [0, pad], [0, pad], [0, 0]])
                        inputs[j] = tf.concat([tmp, layer], 3)
                    elif layer.shape[1] < inputs[j].shape[1]:
                        tmp = tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])
                        inputs[j] = tf.concat([inputs[j], tmp], 3)
                    else:
                        inputs[j] = tf.concat([inputs[j], layer], 3)
                else:
                    inputs[j] = layer
                    getinput[j] = True

        # give last layer a name
        last_layer = tf.identity(layer, name="last_layer" + str(self.blocks))
        return last_layer

    def _loss(self, labels, logits):
        """
          Args:
            logits: Logits from softmax.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [self.batch_size]

          Returns:
            Loss tensor of type float.
          """
        # Reshape the labels into a dense Tensor of shape [self.batch_size, self.NUM_CLASSES].
        sparse_labels = tf.reshape(labels, [self.batch_size, 1])
        indices = tf.reshape(tf.range(self.batch_size), [self.batch_size, 1])
        concated = tf.concat([indices, sparse_labels], 1)
        dense_labels = tf.sparse_to_dense(concated,
                                          [self.batch_size, self.NUM_CLASSES],
                                          1.0, 0.0)
        # Calculate loss.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=dense_labels, logits=logits))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = cross_entropy + l2 * self.weight_decay
        return loss

    def _train(self, global_step, loss):
        # Variables that affect learning rate.
        num_batches_per_epoch = self.train_num / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        self.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = tf.train.MomentumOptimizer(lr, self.momentum_rate, use_nesterov=True).minimize(loss,global_step=global_step)
        return train_op,lr

    def evaluate(self, graph_part, cell_list, pre_block=[], is_bestNN=False, update_pre_weight=False):
        '''Method for evaluate the given network.
        Args:
            graph_part: The topology structure of the network given by adjacency table
            cell_list: The configuration of this network for each node in graph_part.
            pre_block: The pre-block structure, every block has two parts: graph_part and cell_list of this block.
            is_bestNN: Symbol for indicating whether the evaluating network is the best network of this round, default False.
            update_pre_weight: Symbol for indicating whether to update previous blocks' weight, default by False.
        Returns:
            Accuracy'''
        #TODO function is still too long, need to be splited
        if self.train_num < self.batch_size:
            print("Wrong! The data added in train dataset is smaller than batch size!")
            self.add_data(self.batch_size - self.train_num)
            print("Default add batch size picture to the train dataset.")
        self.block_num = len(pre_block)

        """Train CIFAR-10 for a number of steps."""
        with tf.Session() as sess:
            global_step = tf.Variable(0, trainable=False)
            train_flag = tf.placeholder(tf.bool)

            # if it got previous blocks
            if self.block_num > 0:
                new_saver = tf.train.import_meta_graph(
                    os.path.join(self.model_path, 'model_block' + str(self.blocks - 1) + '.meta'))
                new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name("input:0")
                labels = graph.get_tensor_by_name("label:0")
                input = graph.get_tensor_by_name("last_layer" + str(self.block_num - 1) + ":0")
                # a pooling later for every block
                input = self._makepool(input, ('pool', 'max', 2))
                # only when there's not so many network in the pool will we update the previous blocks' weight
                if not update_pre_weight:
                    input = tf.stop_gradient(input, name="stop_gradient")
            # if it's the first block
            else:
                x = tf.placeholder(tf.float32, [self.batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, 3], name='input')
                labels = tf.placeholder(tf.int32, [self.batch_size], name="label")
                input = x

            logits = self._inference(input, graph_part, cell_list, train_flag)
            # softmax
            logits = tf.reshape(logits, [self.batch_size, -1])
            with tf.variable_scope('lastdense' + str(self.block_num)) as scope:
                weights = tf.get_variable('weights' + str(self.block_num), shape=[logits.shape[-1], self.NUM_CLASSES],
                                          initializer=tf.truncated_normal_initializer(stddev=0.04))
                biases = tf.get_variable('biases' + str(self.block_num), shape=[self.NUM_CLASSES],
                                         initializer=tf.constant_initializer(0.0))
                logits = tf.add(tf.matmul(logits, weights), biases, name=scope.name)

            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            loss = self._loss(labels, logits)
            train_op,lr = self._train(global_step, loss)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())
            # Start running operations on the Graph.
            sess.run(tf.global_variables_initializer())

            for ep in range(self.epoch):
                for step in range(self.max_steps):
                    start_time = time.time()
                    batch_x = self.train_data[step * self.batch_size:step * self.batch_size + self.batch_size]
                    batch_y = self.train_label[step * self.batch_size:step * self.batch_size + self.batch_size]
                    _, loss_value,lrt = sess.run([train_op, loss,lr],
                                             feed_dict={x: batch_x, labels: batch_y, train_flag: True})

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 100 == 0:
                        format_str = ('step %d, loss = %.2f, learning rate=%.4f (%.3f sec)')
                        print(format_str % (step, loss_value,lrt, float(time.time() - start_time)*100))

                if is_bestNN:  # Save the model
                    saver.save(sess, self.model_path + 'model_block' + str(self.blocks))

                # Start the queue runners.
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                    num_iter = self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // self.batch_size
                    true_count = 0  # Counts the number of correct predictions.
                    total_sample_count = num_iter * self.batch_size
                    step = 0
                    start_time = time.time()
                    while step < num_iter and not coord.should_stop():
                        batch_x = self.valid_data[step * self.batch_size:step * self.batch_size + self.batch_size]
                        batch_y = self.valid_label[step * self.batch_size:step * self.batch_size + self.batch_size]
                        predictions = sess.run([top_k_op], feed_dict={x: batch_x, labels: batch_y, train_flag: True})
                        true_count += np.sum(predictions)
                        step += 1

                    precision = true_count / total_sample_count
                    print('%d epoch: precision @ 1 = %.3f, cost time %.3f' % (ep,precision, float(time.time() - start_time)))

                except Exception as e:
                    coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

        return precision

    def add_data(self, add_num=0):
        if self.train_num + add_num > self.NUM_EXAMPLES_FOR_TRAIN or add_num < 0:
            add_num = self.NUM_EXAMPLES_FOR_TRAIN - self.train_num
            self.train_num = self.NUM_EXAMPLES_FOR_TRAIN
            print('Warning! Add number has been changed to ', add_num, ', all data is loaded.')
        else:
            self.train_num += add_num
        # print('************A NEW ROUND************')
        self.max_steps = self.train_num // self.batch_size
        return 0


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    eval = Evaluator()
    eval.add_data(50000)
    # graph_full = [[1], [2], [3], []]
    # cell_list = [('conv', 64, 5, 'relu'), ('pooling', 'max', 3), ('conv', 64, 5, 'relu'), ('pooling', 'max', 3)]
    # cell_list = [cell_list]
    # e=eval.evaluate(graph_full,cell_list[-1])#,is_bestNN=True)
    # print(e)
    # cellist=[('conv', 128, 1, 'relu'), ('conv', 32, 1, 'relu'), ('conv', 256, 1, 'relu'), ('pooling', 'max', 2), ('pooling', 'global', 3), ('conv', 32, 1, 'relu')]
    # cellist=[('pooling', 'global', 2), ('pooling', 'max', 3), ('conv', 21, 32, 'leakyrelu'), ('conv', 16, 32, 'leakyrelu'), ('pooling', 'max', 3), ('conv', 16, 32, 'leakyrelu')]

    graph_full = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17],
                  []]
    cell_list = [('conv', 64, 3, 'relu'), ('conv', 64, 3, 'relu'), ('pooling', 'max', 2), ('conv', 128, 3, 'relu'),
                 ('conv', 128, 3, 'relu'), ('pooling', 'max', 2), ('conv', 256, 3, 'relu'),
                 ('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('pooling', 'max', 2),
                 ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
                 ('pooling', 'max', 2), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
                 ('conv', 512, 3, 'relu'), ('dense', [4096, 4096, 1000], 'relu')]
    cell_list = [cell_list]
    # pre_block=[graph_full, cell_list[-1]]
    e = eval.evaluate(graph_full, cell_list[-1])  # , update_pre_weight=True)
    # e=eval.train(network.graph_full,cellist)
    print(e)
