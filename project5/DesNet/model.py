# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt

from datasets import *
from ops import *


class DesNet(object):

    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.build_model(flags)


    def descriptor(self, inputs, is_training=True, reuse=False):
        ####################################################
        # Define network structure for descriptor.
        # Recommended structure:
        # conv1: channel 64 kernel 4*4 stride 2
        # conv2: channel 128 kernel 2*2 stride 1
        # fc: channel output 1
        # conv1 - bn - relu - conv2 - bn - relu -fc
        ####################################################
        with tf.variable_scope('des', reuse=reuse):

            # layer 0:
            h0 = conv2d(inputs, 64, kernal=(4, 4), strides=(2, 2), name='d_conv_0')
            h0 = batch_norm(h0, train=is_training, name="d_bn_0")
            h0 = leaky_relu(h0)

            # layer 1:
            h1 = conv2d(h0, 128, kernal=(2, 2), strides=(1, 1), name='d_conv_1')
            h1 = batch_norm(h1, train=is_training, name="d_bn_1")
            h1 = leaky_relu(h1)

            # layer 2:
            h2 = tf.reshape(h1, [-1, 32*32*128])
            h2 = linear(h2, 1, 'd_ln_2')

            return h2


    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        def cond(i, samples):
            return tf.less(i, flags.T)

        def body(i, samples):
            noise = tf.random_normal(shape=tf.shape(samples), name='noise')
            syn_res = self.descriptor(samples, is_training=True, reuse=True)
            grad = tf.gradients(syn_res, samples, name='grad_des')[0]
            samples = samples - 0.5 * flags.delta**2 * (samples / flags.ref_sig**2 - grad)
            samples = samples + flags.delta * noise
            return tf.add(i, 1), samples

        i = tf.constant(0)
        _, samples = tf.while_loop(cond, body, [i, samples])

        return samples


    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        self.original_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])
        self.synthesized_images = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3])

        self.m_original = self.descriptor(self.original_images)
        self.m_synthesized = self.descriptor(self.synthesized_images, reuse=True)
        self.train_loss = tf.subtract(tf.reduce_mean(self.m_synthesized), tf.reduce_mean(self.m_original))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'des' in var.name]

        self.train_op = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.train_loss, var_list=self.d_vars)
        self.sampling_op = self.Langevin_sampling(self.synthesized_images, flags)


    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)

        saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

        print(" Start training ...")

        des_loss_total = []

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        mean = np.mean(train_data, axis=(1, 2))
        Y_start = np.ones_like(train_data)
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                Y_start[i, :, :, j] = Y_start[i, :, :, j] * mean[i, j]

        for epoch in range(flags.epoch):

            Y_evolve = self.sess.run([self.sampling_op], feed_dict={self.synthesized_images: Y_start})[0]

            _ = self.sess.run([self.train_op], feed_dict={self.synthesized_images: Y_evolve, self.original_images: train_data})

            den_loss = self.sess.run([self.train_loss], feed_dict={self.synthesized_images: Y_evolve, self.original_images: train_data})

            print("Epoch: {:2d} den_loss: {:.8f}".format(epoch, den_loss[0]))

            des_loss_total.append(den_loss[0])

            if np.mod(epoch, 50) == 2 or epoch == flags.epoch - 1:
                
                save_images(Y_evolve, './{}/train_{:02d}.png'.format(self.sample_dir, epoch))


        # Sub-Question (2):
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}
        mplt.rc('font', **font)
        plt.figure()
        plt.title('Training Loss of DesNet')
        plt.plot(des_loss_total, linewidth=2.0, label='des_loss')
        plt.ylabel('Loss')
        plt.xlabel('Number of epochs')
        plt.grid()
        plt.show()