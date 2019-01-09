from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import range
from skimage import io

from ops import *
from utils import *
import random
import numpy as np
import cv2


class VAE(object):
    def __init__(self, sess, image_size=28,
                 batch_size=100, sample_size=100, output_size=28,
                 z_dim=20, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            image_size: The size of input image.
            batch_size: The size of batch. Should be specified before training.
            sample_size: (optional) The size of sampling. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [28]
            z_dim: (optional) Dimension of latent vectors. [5]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()


    def encoder(self, image, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # TODO: Define encoder network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            # The output of encoder network should have two parts:
            # A mean vector and a log(std) vector. Both of them have
            # the same dimension with latent vector z.
            #######################################################

            # initializers:
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.0)

            # layer 0:
            x = tf.reshape(image, [-1, self.image_size*self.image_size])
            w0 = tf.get_variable('w0', [x.get_shape()[1], 512], initializer=w_init)
            b0 = tf.get_variable('b0', [512], initializer=b_init)
            h0 = tf.matmul(x, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, 0.9)

            # layer 1:
            w1 = tf.get_variable('w1', [h0.get_shape()[1], 512], initializer=w_init)
            b1 = tf.get_variable('b1', [512], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, 0.9)

            # layer 2:
            w2 = tf.get_variable('w2', [h1.get_shape()[1], self.z_dim*2], initializer=w_init)
            b2 = tf.get_variable('b2', [self.z_dim*2], initializer=b_init)
            h2 = tf.matmul(h1, w2) + b2

            # output mean and sigma:
            mean = h2[:, :self.z_dim]
            sigma = 1e-6 + tf.nn.softplus(h2[:, self.z_dim:])

        return mean, sigma

        #######################################################
        #                   end of your code
        #######################################################


    def decoder(self, z, reuse=False, train=True):
        with tf.variable_scope("decoder", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. To make the
            # output pixel values in [0,1], add a sigmoid layer before
            # the output. Also use batch normalization layer after
            # deconv layer, and use 'train' argument to indicate the
            # mode of bn layer. Note that when sampling images using
            # trained model, you need to set train='False'.
            #######################################################

            # initializers:
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.0)

            # layer 0:
            w0 = tf.get_variable('w0', [z.get_shape()[1], 512], initializer=w_init)
            b0 = tf.get_variable('b0', [512], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, 0.9)

            # layer 1:
            w1 = tf.get_variable('w1', [h0.get_shape()[1], 512], initializer=w_init)
            b1 = tf.get_variable('b1', [512], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, 0.9)

            # layer 2:
            w2 = tf.get_variable('w2', [h1.get_shape()[1], 512], initializer=w_init)
            b2 = tf.get_variable('b2', [512], initializer=b_init)
            h2 = tf.matmul(h1, w2) + b2
            h2 = tf.nn.elu(h2)
            h2 = tf.nn.dropout(h2, 0.9)

            # layer 3:
            w3 = tf.get_variable('w3', [h2.get_shape()[1], 28*28], initializer=w_init)
            b3 = tf.get_variable('b3', [28*28], initializer=b_init)
            y = tf.sigmoid(tf.matmul(h2, w3) + b3)
            y = tf.reshape(y, [-1, self.image_size, self.image_size, 1])

        return y
        
        #######################################################
        #                   end of your code
        #######################################################


    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of VAE. For input,
        # you need to define it as placeholders. Remember loss
        # term has two parts: reconstruction loss and KL divergence
        # loss. Save the loss as self.loss. Use the
        # reparameterization trick to sample z.
        #######################################################

        # define model inputs:
        self.image = tf.placeholder(name='image',
                                    dtype=tf.float32,
                                    shape=[self.batch_size, self.image_size, self.image_size, 1])

        # apply reparameterization trick to sample z:
        mean, sigma = self.encoder(self.image, reuse=False, train=True)
        z = mean + sigma * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)

        # decode z:
        y = self.decoder(z, reuse=False, train=True)
        y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # compute loss:
        image_vec = tf.reshape(self.image, [-1, self.image_size * self.image_size * 1])
        y_vec = tf.reshape(y, [-1, self.image_size * self.image_size * 1])
        marginal_likelihood = tf.reduce_sum(image_vec * tf.log(y_vec) + (1 - image_vec) * tf.log(1 - y_vec), 1)
        KL_divergence = 0.5 * tf.reduce_sum(1 + tf.log(1e-8 + tf.square(sigma)) - tf.square(mean) - tf.square(sigma), 1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)
        ELBOL = marginal_likelihood + KL_divergence

        self.loss = -ELBOL
        self.image_reconstruct = y

        #######################################################
        #                   end of your code
        #######################################################
        self.saver = tf.train.Saver()


    def train(self, config):
        """Train VAE"""
        # load MNIST dataset
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        data = mnist.train.images
        data = data.astype(np.float32)
        data_len = data.shape[0]
        data = np.reshape(data, [-1, 28, 28, 1])

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sample_dir = os.path.join(config.sample_dir, config.dataset)
        if not os.path.exists(config.sample_dir):
            os.mkdir(config.sample_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)


        for epoch in range(config.epoch):
            batch_idxs = min(data_len, config.train_size) // config.batch_size
            for idx in range(0, batch_idxs):
                counter += 1
                batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size, :]
                #######################################################
                # TODO: Train your model here, print the loss term at
                # each training step to monitor the training process.
                # Print reconstructed images and sample images every
                # config.print_step steps. Sample z from standard normal
                # distribution for sampling images. You may use function
                # save_images in utils.py to save images.
                #######################################################

                _, loss_total, image_reconstruct = self.sess.run([optim, self.loss, self.image_reconstruct],
                                                                 feed_dict={self.image: batch_images})

                print("Epoch: {:2d} {:4d}/{:4d} loss: {:.8f}".format(epoch,
                                                                     idx,
                                                                     batch_idxs,
                                                                     loss_total))

                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    figure = np.stack([batch_images[0:14, :, :, :],
                                      image_reconstruct[0:14, :, :, :]],
                                      axis=1)
                    figure = stack_images(figure)
                    figure = np.clip(figure*255, 0, 255).astype('uint8')

                    cv2.imshow("", figure[:, :, 0])
                    cv2.waitKey(1)

                    save_images(image_reconstruct,
                                image_manifold_size(image_reconstruct.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)

            
    def save(self, checkpoint_dir, step):
        model_name = "mnist.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
