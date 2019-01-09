import tensorflow as tf
import os

from model import DesNet

flags = tf.app.flags
flags.DEFINE_string('dataset_name', 'egret', 'Folder name which stored in ./Image/dataset_name')
flags.DEFINE_string('output_dir', 'egret', 'pattern of filename of input images [*]')

# hyper parameters for learning process
flags.DEFINE_integer('batch_size', 7, 'Batch size for training')
flags.DEFINE_integer('epoch', 1000, 'Number of epoch for training')
flags.DEFINE_integer('image_size', 64, 'image size of training images')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
flags.DEFINE_float('beta1', 0.5, 'Momentum')

# hyper parameters for Langevin dynamics, no need to change
flags.DEFINE_integer('T', 30, 'Number of Langevin iterations,')
flags.DEFINE_float('delta', 0.3, 'Langevin step size')
flags.DEFINE_float('ref_sig', 50, 'Standard deviation for reference gaussian distribution')

FLAGS = flags.FLAGS


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        models = DesNet(sess, FLAGS)

    models.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
