from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mplt

from ops import *
from datasets import *


class GenNet(object):

    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.delta = config.delta
        self.sigma = config.sigma
        self.sample_steps = config.sample_steps
        self.z_dim = config.z_dim

        self.num_epochs = config.num_epochs
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

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

        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)
        self.build_model()


    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        with tf.variable_scope("gen", reuse=reuse):

            # compute filter sizes:
            s4 = self.image_size
            s3 = int(math.ceil(s4 / 2.0))
            s2 = int(math.ceil(s3 / 2.0))
            s1 = int(math.ceil(s2 / 2.0))
            s0 = int(math.ceil(s1 / 2.0))

            # layer 0:
            h0 = linear(inputs, 512*s0*s0, 'g_ln_0')
            h0 = tf.reshape(h0, [-1, s0, s0, 512])
            h0 = batch_norm(h0, train=is_training, name="g_bn_0")
            h0 = leaky_relu(h0)

            # layer 1:
            h1 = deconv2d(h0, [self.batch_size, s1, s1, 256], name='g_deconv2d_1')
            h1 = batch_norm(h1, train=is_training, name="g_bn_1")
            h1 = leaky_relu(h1)

            # layer 2:
            h2 = deconv2d(h1, [self.batch_size, s2, s2, 128], name='g_deconv2d_2')
            h2 = batch_norm(h2, train=is_training, name="g_bn_2")
            h2 = leaky_relu(h2)

            # layer 3:
            h3 = deconv2d(h2, [self.batch_size, s3, s3, 64], name='g_deconv2d_3')
            h3 = batch_norm(h3, train=is_training, name="g_bn_3")
            h3 = leaky_relu(h3)

            # layer 4:
            h4 = deconv2d(h3, [self.batch_size, s4, s4, 3], name='g_deconv2d_4')
            h4 = tf.nn.tanh(h4)


            return h4


    def langevin_dynamics(self, z):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        def cond(step_cur, z):
            return tf.less(step_cur, self.sample_steps)

        def body(step_cur, z):
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - gen_res), axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_des')[0]
            z = z + self.delta * noise - 0.5 * self.delta * self.delta * (z + grad)
            step_cur = tf.add(step_cur, 1)
            return step_cur, z

        step_cur = tf.constant(0)
        _, z = tf.while_loop(cond, body, [step_cur, z])
        return z


    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        g_res = self.generator(self.z)
        self.image_reconstruct = g_res
        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - g_res))

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.infer_op = self.langevin_dynamics(self.z)
        self.train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.gen_loss, var_list=self.g_vars)
        

    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.graph.finalize()

        print('Start training ...')

        z = np.random.normal(0, 1.0, (self.batch_size, self.z_dim)).astype(np.float32)

        gen_loss_total = []

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
        for epoch in range(self.num_epochs):

            z = self.infer_op.eval({self.z: z, self.obs: train_data})

            _, image_reconstruct = self.sess.run([self.train_op, self.image_reconstruct], feed_dict={self.z: z, self.obs: train_data})

            gen_loss = self.gen_loss.eval({self.z: z, self.obs: train_data})

            print("Epoch: {:2d} gen_loss: {:.8f}".format(epoch, gen_loss))

            gen_loss_total.append(gen_loss)

            if np.mod(epoch, self.log_step) == 2 or epoch == self.num_epochs - 1:
                
                save_images(image_reconstruct, './{}/train_{:02d}.png'.format(self.sample_dir, epoch))


        # Sub-Question (2):
        z = np.random.normal(0, 1.0, (self.batch_size, self.z_dim)).astype(np.float32)
        image_reconstruct = self.image_reconstruct.eval({self.z: z})
        save_images(image_reconstruct, './{}/random.png'.format(self.sample_dir))


        # Sub-Question (3):
        z1, z2 = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
        z1 = np.array(z1.reshape((-1, 1)))
        z2 = np.array(z2.reshape((-1, 1)))
        z = np.concatenate((z1, z2), axis=1)
        z = np.concatenate((z, np.zeros((2, 2))), axis=0)
        i_max = int(z.shape[0] / self.batch_size)
        for i in range(i_max):
            z_cur = z[i*self.batch_size : (i+1)*self.batch_size]
            image_reconstruct = self.image_reconstruct.eval({self.z: z_cur})
            save_images(image_reconstruct, './{}/linear_{:02d}.png'.format(self.sample_dir, i))


        # Sub-Question (4):
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}
        mplt.rc('font', **font)
        plt.figure()
        plt.title('Training Loss of GenNet')
        plt.plot(gen_loss_total, linewidth=2.0, label='gen_loss')
        plt.ylabel('Loss')
        plt.xlabel('Number of epochs')
        plt.grid()
        plt.show()