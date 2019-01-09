import os
import scipy.misc
import numpy as np

from model_vae import VAE
from utils import pp, visualize, to_json
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.00001, "Uper bound of learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 28, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for tcd raining, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        vae = VAE(sess,
                      image_size=FLAGS.image_size,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      c_dim=FLAGS.c_dim,
                      dataset_name=FLAGS.dataset,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)

        if FLAGS.is_train:
            vae.train(FLAGS)
        else:
            vae.load(FLAGS.checkpoint_dir)

if __name__ == '__main__':
    tf.app.run()
