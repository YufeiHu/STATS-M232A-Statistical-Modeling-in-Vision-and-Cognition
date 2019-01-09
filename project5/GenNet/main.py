import argparse

import tensorflow as tf

from GenNet import GenNet


def main():
    parser = argparse.ArgumentParser()

    # GConvNet hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=2)

    # training hyper-parameters, no need to change
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=11)
    parser.add_argument('--g_lr', type=float, default=0.001)  # learning rate
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--sigma', type=float, default=0.3)

    # langevin hyper-parameters
    parser.add_argument('--delta', type=float, default=0.3)  # step size of Langevin sampling
    parser.add_argument('--sample_steps', type=int, default=30)  # sample steps of Langevin sampling

    # misc
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='.\output')
    parser.add_argument('--category', type=str, default='lion_tiger')
    parser.add_argument('--data_path', type=str,
                        default='./Image/')
    parser.add_argument('--log_step', type=int, default=20)

    opt = parser.parse_args()

    with tf.Session() as sess:
        model = GenNet(sess, opt)
        model.train()


if __name__ == '__main__':
    main()
