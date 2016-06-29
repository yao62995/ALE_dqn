#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import tensorflow as tf

from ale_util import logger


class DLNetwork(object):
    def __init__(self, game_name, action_num, args):
        self.model_dir = args.saved_model_dir
        if self.model_dir == "":
            self.model_dir = "./saved_networks/%s" % game_name
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.actions = action_num
        self.frame_freq_num = args.frame_seq_num
        # tensorflow session
        # self.session = tf.InteractiveSession()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
        # set tensorflow variable
        self.tf_val = dict()
        if args.device == "cpu":
            self.device = "/cpu:0"
        else:
            self.device = "/gpu:%d" % args.gpu
        with tf.device(self.device):
            self.tf_val["in"], self.tf_val["out"] = DLNetwork.create_cnn_net_v2(self.actions, self.frame_freq_num)
            # set loss function
            self.loss_function(args)
        # set model saver
        self.saver = tf.train.Saver(max_to_keep=None)
        # register all variable
        self.session.run(tf.initialize_all_variables())
        # load model if exist
        model_file = None
        if args.model_file != "":
            model_file = args.model_file
        self.restore_model(model_file=model_file)

    def loss_function(self, args):
        # with tf.device('/cpu:0'):
        self.tf_val["act"] = tf.placeholder("float", [None, self.actions])
        self.tf_val["target"] = tf.placeholder("float", [None])
        predict_act = tf.reduce_sum(tf.mul(self.tf_val["out"], self.tf_val["act"]), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.tf_val["target"] - predict_act))

        if args.optimizer == "rmsprop":
            self.tf_val["train"] = tf.train.RMSPropOptimizer(args.learn_rate,
                                                             decay=args.decay_rate,
                                                             momentum=args.momentum).minimize(cost)
        elif args.optimizer == "adam":
            self.tf_val["train"] = tf.train.AdamOptimizer(args.learn_rate).minimize(cost)
        elif args.optimizer == "sgd":
            self.tf_val["train"] = tf.train.GradientDescentOptimizer(args.learn_rate).minimize(cost)

    def fit(self, state, action, target):
        self.tf_val["train"].run(session=self.session, feed_dict={
            self.tf_val["in"]: state,
            self.tf_val["act"]: action,
            self.tf_val["target"]: target,
        })

    def predict(self, batch_states):
        predict_out = self.tf_val["out"].eval(session=self.session, feed_dict={self.tf_val["in"]: batch_states})
        return predict_out

    def restore_model(self, model_file=None):
        if model_file is not None:
            model_file_path = "%s/%s" % (self.model_dir, model_file)
            self.saver.restore(self.session, model_file_path)
            logger.info("Successfully loaded: %s" % model_file_path)
        else:
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                logger.info("Successfully loaded: %s" % checkpoint.model_checkpoint_path)
            else:
                logger.warn("Could not find old network weights")

    def save_model(self, prefix, global_step):
        self.saver.save(self.session, self.model_dir + "/" + prefix, global_step=global_step)

    @staticmethod
    def weight_variable(shape, stddev=0.01):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    @staticmethod
    def create_cnn_net(actions, frame_freq_num):
        # with tf.device('/gpu:1'):
        W_conv1 = DLNetwork.weight_variable([8, 8, frame_freq_num, 32])
        b_conv1 = DLNetwork.bias_variable([32])

        W_conv2 = DLNetwork.weight_variable([5, 5, 32, 64])
        b_conv2 = DLNetwork.bias_variable([64])

        W_conv3 = DLNetwork.weight_variable([3, 3, 64, 64])
        b_conv3 = DLNetwork.bias_variable([64])

        W_fc1 = DLNetwork.weight_variable([256, 128])
        b_fc1 = DLNetwork.bias_variable([128])

        W_fc2 = DLNetwork.weight_variable([128, actions])
        b_fc2 = DLNetwork.bias_variable([actions])

        # input layer
        _input = tf.placeholder("float", [None, 80, 80, frame_freq_num])

        # hidden layer
        h_conv1 = DLNetwork.conv2d(_input, W_conv1, stride=4) + b_conv1
        h_act1 = tf.nn.relu(h_conv1)
        h_pool1 = DLNetwork.max_pool_2x2(h_act1)

        h_conv2 = DLNetwork.conv2d(h_pool1, W_conv2, stride=2) + b_conv2
        h_act2 = tf.nn.relu(h_conv2)
        h_pool2 = DLNetwork.max_pool_2x2(h_act2)

        h_conv3 = DLNetwork.conv2d(h_pool2, W_conv3, stride=1) + b_conv3
        h_act3 = tf.nn.relu(h_conv3)
        h_pool3 = DLNetwork.max_pool_2x2(h_act3)

        h_flat1 = tf.reshape(h_pool3, [-1, 256])

        h_fc1 = tf.matmul(h_flat1, W_fc1) + b_fc1
        h_act4 = tf.nn.relu(h_fc1)

        _output = tf.matmul(h_act4, W_fc2) + b_fc2
        return _input, _output

    @staticmethod
    def create_cnn_net_v2(actions, frame_freq_num):
        # with tf.device('/gpu:1'):
        W_conv1 = DLNetwork.weight_variable([8, 8, frame_freq_num, 32])
        b_conv1 = DLNetwork.bias_variable([32])

        W_conv2 = DLNetwork.weight_variable([5, 5, 32, 64])
        b_conv2 = DLNetwork.bias_variable([64])

        W_conv3 = DLNetwork.weight_variable([3, 3, 64, 32])
        b_conv3 = DLNetwork.bias_variable([32])

        W_fc1 = DLNetwork.weight_variable([800, 128])
        b_fc1 = DLNetwork.bias_variable([128])

        W_fc2 = DLNetwork.weight_variable([128, actions])
        b_fc2 = DLNetwork.bias_variable([actions])

        # input layer
        _input = tf.placeholder("float", [None, 80, 80, frame_freq_num])

        # hidden layer
        h_conv1 = DLNetwork.conv2d(_input, W_conv1, stride=4) + b_conv1
        h_act1 = tf.nn.relu(h_conv1)
        # h_pool1 = DLNetwork.max_pool_2x2(h_act1)

        h_conv2 = DLNetwork.conv2d(h_act1, W_conv2, stride=2) + b_conv2
        h_act2 = tf.nn.relu(h_conv2)
        # h_pool2 = DLNetwork.max_pool_2x2(h_act2)

        h_conv3 = DLNetwork.conv2d(h_act2, W_conv3, stride=2) + b_conv3
        h_act3 = tf.nn.relu(h_conv3)
        # h_pool3 = DLNetwork.max_pool_2x2(h_act3)

        h_flat1 = tf.reshape(h_act3, [-1, 800])

        h_fc1 = tf.matmul(h_flat1, W_fc1) + b_fc1
        h_act4 = tf.nn.relu(h_fc1)

        _output = tf.matmul(h_act4, W_fc2) + b_fc2
        return _input, _output

    @staticmethod
    def create_mlp_net(actions):

        W_fc1 = DLNetwork.weight_variable([256, 512], stddev=1.0)
        b_fc1 = DLNetwork.bias_variable([512])

        W_fc2 = DLNetwork.weight_variable([512, 128], stddev=1.0)
        b_fc2 = DLNetwork.bias_variable([128])

        W_fc3 = DLNetwork.weight_variable([128, 32], stddev=1.0)
        b_fc3 = DLNetwork.bias_variable([32])

        W_fc4 = DLNetwork.weight_variable([32, actions])
        b_fc4 = DLNetwork.bias_variable([actions])

        _input = tf.placeholder("float", [None, 8, 8, 4])

        h_flat1 = tf.reshape(_input, [-1, 256])  # flatten [None, 4*8*8]

        h_fc1 = tf.matmul(h_flat1, W_fc1) + b_fc1
        h_act1 = tf.nn.relu(h_fc1)
        h_dropout1 = tf.nn.dropout(h_act1, 0.2)

        h_fc2 = tf.matmul(h_dropout1, W_fc2) + b_fc2
        h_act2 = tf.nn.relu(h_fc2)
        h_dropout2 = tf.nn.dropout(h_act2, 0.2)

        h_fc3 = tf.matmul(h_dropout2, W_fc3) + b_fc3
        h_act3 = tf.nn.relu(h_fc3)
        h_dropout3 = tf.nn.dropout(h_act3, 0.2)

        h_fc4 = tf.matmul(h_dropout3, W_fc4) + b_fc4
        _output = tf.nn.relu(h_fc4)
        return _input, _output
