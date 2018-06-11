import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import argparse
import os

sns.set_style("whitegrid")

parser = argparse.ArgumentParser(description="Network for image classification")
parser.add_argument('--data_dir', default='tem/data', help='Directory for training data')
parser.add_argument('--result_dir', default='tem/result')
parser.add_argument('--model_dir', default='model/', help='the place of saving networks parameters')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--print_loss', default=10)
parser.add_argument('--plot_loss', default=100)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--n_iterations', default=10000, type=int)
args = parser.parse_args()

w_init = tf.random_normal_initializer(stddev=0.01)
def network(x):
    layers1 = tf.layers.conv2d(x, 32, 3, 1, padding='same', activation=tf.nn.relu, kernel_initializer=w_init)
    layers2 = tf.layers.conv2d(layers1, 62, 3, 1, padding='same', activation=tf.nn.relu, kernel_initializer=w_init)
    layers2_flatten = tf.contrib.layers.flatten(layers2)
    layers3 = tf.layers.dense(layers2_flatten, 200, activation=tf.nn.relu, kernel_initializer=w_init)
    output = tf.layers.dense(layers3, 10, activation=tf.nn.tanh, kernel_initializer=w_init)
    return output
def training():
    input_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    label_y = tf.placeholder(tf.float32, [None, 10])
    output_y = network(input_x)
    loss = tf.reduce_sum(tf.square(label_y-output_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    init_all_v = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_all_v)
    saver = load_model(sess)
    mnist = read_data_sets(args.data_dir, one_hot=True)
    print('start training')
    plot_loss = []
    for i in range(args.n_iterations):
        batch_x, batch_y = mnist.train.next_batch(args.batch_size)
        batch_x = batch_x.reshape([args.batch_size, 28, 28, 1])
        y = np.zeros([args.batch_size, 10])
        for j in range(args.batch_size):
            k = batch_y[j].astype(np.int)
            y[j, k] = 1.
        batch_y = y
        d_loss, _ = sess.run([loss, optimizer], feed_dict={input_x:batch_x, label_y:batch_y})
        plot_loss.append(d_loss)

        if i % args.print_loss == 0 and i > 0:
            print('Iteration is : %d, Loss is: %f' % (i, d_loss))
        if i % args.plot_loss == 0 and i > 0:
            plt.figure(figsize=(6*1.1618, 6))
            plt.plot(range(len(plot_loss)), plot_loss)
            plt.xlabel('iteration times')
            plt.ylabel('lost')
            plt.show()
        if i % 500 == 0 and i > 0:
            save_model(saver, sess, i)
def save_model(saver, sess, step):
    saver.save(sess, os.path.join(args.model_dir, "classification"), global_step=step)
    # saver.save(sess, os.path.join(args.network_dir, "gan_network"), global_step=step)
def load_model(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(args.model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find any old weights!")
    return saver

def main(_):
    training()
if __name__ == "__main__":
    tf.app.run()









