#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/16 下午6:17
# @Author  : niuliangtao
# @Site    :
# @File    : ranknetTest1.py
# @Software: PyCharm

from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import random

from ReadData import Employer

tw = random.random(size=(10, 10))


class EmployerTrain:
    def __init__(self):
        self.a = None
        self.batch_size = 100

        # Parameters
        self.learning_rate = 0.01
        self.training_epochs = 1
        self.display_step = 1
        self.train_data = None
        self.step = 0

    def init_data(self):

        work = Employer()
        work.read_data()
        work.init_data()
        self.train_data = work.train_data

    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation

        layer_1 = tf.nn.dropout(x, 0.5)
        layer_1 = tf.add(tf.matmul(layer_1, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.nn.dropout(layer_1, 0.5)
        layer_2 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

        layer_3 = tf.nn.dropout(layer_2, 0.5)
        layer_3 = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
        # layer_3 = tf.nn.relu(layer_3)

        # return layer_1
        return layer_3

    def get_train_data(self, batch_size=32):
        # 生成的数据特征维数为10,lable为前5个维度的特征 * 2 + 后五个维度的特征 * 3得到
        xs = []
        ys = []

        for i in range(0, batch_size):
            x = random.random(size=(10))
            y = np.matmul(x, tw)
            xs.append(x)
            ys.append(y)

        # print(self.train_data.size)
        if (self.step + batch_size) >= self.train_data.size / 2800:
            self.step = 0

        xdata = self.train_data.iloc[self.step:self.step + batch_size, :]
        self.step += batch_size

        xs, ys = np.array(xs), np.array(ys)
        xs, ys = xdata.iloc[:, 0:-5], xdata.iloc[:, -5:]
        # xs = np.reshape(xs, batch_size, 10)
        # ys = np.reshape(ys, batch_size, 10)

        return xs, ys

    def train(self):
        # Network Parameters
        n_hidden_0 = 2795  # 输入层单元个数
        n_hidden_1 = 128  # 第一层隐层单元个数
        n_hidden_2 = 64  # 第二层隐层单元个数
        n_output = 5

        # print(self.get_train_data())

        weights = {
            'h1': tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }

        input_label = tf.placeholder(tf.float32, [None, n_output])

        input_net = tf.placeholder(tf.float32, shape=[None, n_hidden_0])

        # -------------------------------------------------------------------------------------------
        # 进入DNN三层网络，得到最后一层隐层,也是softmax层的输入层，可作为用户向量
        output_label = self.multilayer_perceptron(input_net, weights, biases)

        # 计算损失函数
        loss = tf.reduce_mean(tf.square(tf.log(tf.abs(input_label) + 1) - tf.log(tf.abs(output_label) + 1)))

        cost = loss  # / batch_size
        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(100000):
                x, y = self.get_train_data(batch_size=128)
                try:
                    feed_dict = {input_net: x,
                                 input_label: y}
                    _, c = sess.run([optimizer, cost], feed_dict=feed_dict)

                    # if step == 1194:
                    #     print("{0},{1}".format(x, y))
                    #     break
                    if step % 20 == 0:
                        print("step:{0},\tloss:{1}".format(step, c))

                except Exception as e:
                    print("{0},{1},{2}".format(x, y, e))
                    break


if __name__ == '__main__':
    a = EmployerTrain()
    a.init_data()
    a.train()
