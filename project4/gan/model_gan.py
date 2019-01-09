from __future__ import division

import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import cv2
from six.moves import xrange

from ops import *
from utils import *


class DCGAN(object):
    def __init__(self, sess, input_size=28,
                 batch_size=100, sample_num=100, output_size=28,
                 z_dim=100, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          input_size: The size of input image.
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_size = input_size
        self.output_size = output_size

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.build_model()


    def discriminator(self, image, reuse=False, train=True):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            #######################################################
            # TODO: Define discrminator network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            #######################################################

            if reuse:
                scope.reuse_variables()

            # layer 0:
            h0 = conv2d(image, 64, name='d_conv2d_0')
            h0 = lrelu(h0)

            # layer 1:
            h1 = conv2d(h0, 128, name='d_conv2d_1')
            h1 = batch_norm(h1, train=train, name="d_bn_1")
            h1 = lrelu(h1)

            # layer 2:
            h2 = conv2d(h1, 256, name='d_conv2d_2')
            h2 = batch_norm(h2, train=train, name="d_bn_2")
            h2 = lrelu(h2)

            # layer 3:
            h3 = conv2d(h2, 512, name='d_conv2d_3')
            h3 = batch_norm(h3, train=train, name="d_bn_3")
            h3 = lrelu(h3)

            # layer 4:
            h4 = tf.reshape(h3, [self.batch_size, -1])
            h4 = linear(h4, 1, 'd_ln_4')

            # output layer:
            h5 = tf.nn.sigmoid(h4)

            return h5, h4

            #######################################################
            #                   end of your code
            #######################################################


    def generator(self, z, reuse=False, train=True):
        with tf.variable_scope("generator", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. Image scale
            # in DCGAN is [-1, +1], so you need to add a tanh layer
            # before the output. Also use batch normalization layer
            # after deconv layer, and use 'train' argument to indicate
            # the mode of bn layer. Note that when sampling images
            # using trained model, you need to set train='False'.
            #######################################################

            # compute filter sizes:
            s4 = self.output_size
            s3 = int(math.ceil(s4 / 2.0))
            s2 = int(math.ceil(s3 / 2.0))
            s1 = int(math.ceil(s2 / 2.0))
            s0 = int(math.ceil(s1 / 2.0))

            # layer 0:
            h0 = linear(z, 512*s0*s0, 'g_ln_0')
            h0 = tf.reshape(h0, [-1, s0, s0, 512])
            h0 = batch_norm(h0, name="g_bn_0")
            h0 = tf.nn.relu(h0)

            # layer 1:
            h1 = deconv2d(h0, [self.batch_size, s1, s1, 256], name='g_deconv2d_1')
            h1 = batch_norm(h1, name="g_bn_1")
            h1 = tf.nn.relu(h1)

            # layer 2:
            h2 = deconv2d(h1, [self.batch_size, s2, s2, 128], name='g_deconv2d_2')
            h2 = batch_norm(h2, name="g_bn_2")
            h2 = tf.nn.relu(h2)

            # layer 3:
            h3 = deconv2d(h2, [self.batch_size, s3, s3, 64], name='g_deconv2d_3')
            h3 = batch_norm(h3, name="g_bn_3")
            h3 = tf.nn.relu(h3)

            # layer 4:
            h4 = deconv2d(h3, [self.batch_size, s4, s4, self.c_dim], name='g_deconv2d_4')
            h4 = tf.nn.tanh(h4)

            self.image_reconstruct = h4

            return h4

            #######################################################
            #                   end of your code
            #######################################################


    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of DCGAN. For input,
        # you need to define it as placeholders. Discriminator
        # loss has two parts: cross entropy for real images and
        # cross entropy for fake images generated by generator.
        # Set reuse=True for discriminator when calculating the
        # second cross entropy. Define two different loss terms
        # for discriminator and generator, and save them as
        # self.d_loss and self.g_loss respectively.
        #######################################################

        # define model inputs
        self.image = tf.placeholder(tf.float32,
                                     [self.batch_size, self.output_size, self.output_size, self.c_dim],
                                     name='image')

        self.z = tf.placeholder(tf.float32,
                                [self.batch_size, self.z_dim],
                                name='z')

        # generate and judge
        image_fake = self.generator(self.z)
        probs_real, scores_real = self.discriminator(self.image, reuse=False)
        probs_fake, scores_fake = self.discriminator(image_fake, reuse=True)

        # compute losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores_real, labels=tf.ones_like(probs_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores_fake, labels=tf.zeros_like(probs_fake)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores_fake, labels=tf.ones_like(probs_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        #######################################################
        #                   end of your code
        #######################################################
        # define var lists for generator and discriminator
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, config):
        # create two optimizers for generator and discriminator,
        # and only update the corresponding variables.
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        # load MNIST data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        data = mnist.train.images
        data = data.astype(np.float32)
        data_len = data.shape[0]
        data = np.reshape(data, [-1, 28, 28, 1])
        data = data * 2.0 - 1.0

        counter = 1
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        d_loss_all = []
        g_loss_all = []

        for epoch in xrange(config.epoch):
            batch_idxs = min(data_len, config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size, :]
                #######################################################
                # TODO: Train your model here. Sample hidden z from
                # standard uniform distribution. In each step, run g_optim
                # twice to make sure that d_loss does not go to zero.
                # print the loss terms at each training step to monitor
                # the training process. Print sample images every
                # config.print_step steps.You may use function
                # save_images in utils.py to save images.
                #######################################################

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _ = self.sess.run([d_optim], feed_dict={self.image: batch_images, self.z: batch_z})
                
                # update G network first time
                _ = self.sess.run([g_optim], feed_dict={self.z: batch_z})

                # update G network second time
                _, image_reconstruct = self.sess.run([g_optim, self.image_reconstruct], feed_dict={self.z: batch_z})

                # compute and print losses
                d_loss_print = self.d_loss_real.eval({self.image: batch_images}) + self.d_loss_fake.eval({self.z: batch_z})
                g_loss_print = self.g_loss.eval({self.z: batch_z})
                d_loss_all.append(d_loss_print)
                g_loss_all.append(g_loss_print)
                print("Epoch: {:2d} {:4d}/{:4d} d_loss: {:.8f}, g_loss: {:.8f}".format(epoch,
                                                                                          idx,
                                                                                          batch_idxs,
                                                                                          d_loss_print,
                                                                                          g_loss_print))

                if np.mod(counter, 200) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    figure = np.stack([batch_images[0:14, :, :, :],
                                      image_reconstruct[0:14, :, :, :]],
                                      axis=1)
                    figure = stack_images(figure)
                    figure = (figure + 1.0) / 2.0
                    figure = np.clip(figure*255, 0, 255).astype('uint8')

                    cv2.imshow("", figure[:, :, 0])
                    
                    save_images(image_reconstruct,
                                image_manifold_size(image_reconstruct.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

                    cv2.waitKey(1)

                #######################################################
                #                   end of your code
                #######################################################

                counter += 1
                if np.mod(counter, 500) == 1:
                    self.save(config.checkpoint_dir, counter)

        """
        plt.figure()
        plt.title('Training Loss of GAN')
        plt_d_loss, = plt.plot(d_loss_all, linewidth=2.0, label='d_loss')
        plt_g_loss, = plt.plot(g_loss_all, linewidth=2.0, label='g_loss')
        plt.legend([plt_d_loss, plt_g_loss], ['d_loss', 'g_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Number of iterations')
        plt.grid()
        plt.show()
        """


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_size, self.output_size)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
