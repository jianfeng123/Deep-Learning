import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 选择GPU

sns.set_style('whitegrid')
parser = argparse.ArgumentParser(description="GAN")
parser.add_argument('--data_dir', default='tem/data', help='Data for mnist')
parser.add_argument('--result_dir', default='tem/result', help='Directory of results!')
parser.add_argument('--network_dir', default='model/', help='Directory of model!')
parser.add_argument('--smooth', type=float, default=0.1, help='smooth parameter for gan training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for each iterations')
parser.add_argument('--episodes', type=int, default=100000, help='episodes for training')
parser.add_argument('--d_step', type=int, default=1)
parser.add_argument('--g_step', type=int, default=1)
parser.add_argument('--print_every', type=int, default=200)
parser.add_argument('--plot_every', type=int, default=5000, help="draw the loss curve")
parser.add_argument('--saveI_every', type=int, default=1000, help="how many steps to save a image")
parser.add_argument('--saveW_every', type=int, default=5000, help="how many steps to save whole image")
parser.add_argument('--saveM_every', type=int, default=100, help="how many steps to save model")

args = parser.parse_args()

# np.random.seed(124)
w_init = tf.random_normal_initializer(stddev=0.01)
def Discriminator(x):

    # layer1 = tf.layers.conv2d(x, filters=32, kernel_size=3, kernel_initializer=w_init, padding='same',
    #                           activation=tf.nn.relu)
    # layer2 = tf.layers.conv2d(layer1, filters=64, kernel_size=3, kernel_initializer=w_init, padding='same',
    #                           activation=tf.nn.relu)
    x_flatten = tf.contrib.layers.flatten(x)
    layer1 = tf.layers.dense(x_flatten, 400, activation=tf.nn.relu, kernel_initializer=w_init)
    layer1_drop = tf.layers.dropout(layer1, rate=0.2)
    output_logit = tf.layers.dense(layer1_drop, 1)
    output = tf.nn.sigmoid(output_logit)
    return output, output_logit
def Generator(x):
    # layer1 = tf.layers.conv2d(x, filters=32, kernel_size=3, kernel_initializer=w_init, padding='same',
    #                           activation=tf.nn.relu)
    # layer2 = tf.layers.conv2d(layer1, filters=64, kernel_size=3, kernel_initializer=w_init, padding='same',
    #                           activation=tf.nn.relu)
    x_flatten = tf.contrib.layers.flatten(x)
    layer1 = tf.layers.dense(x_flatten, 200, activation=tf.nn.relu, kernel_initializer=w_init)
    # layer3 = tf.layers.dense(layer2_flatten, 200, activation=tf.nn.relu, kernel_initializer=w_init)
    output = tf.layers.dense(layer1, 28*28, activation=tf.nn.tanh, kernel_initializer=w_init)
    output = tf.reshape(output, [-1, 28, 28, 1])
    return output

def train():
    real_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    noise_input = tf.placeholder(tf.float32, [None, 28, 28, 1])

    with tf.variable_scope('discriminator'):
        real_output, real_logit = Discriminator(real_input)
    with tf.variable_scope('generator'):
        fake_input = Generator(noise_input)                      #相对于判别网络来说
    with tf.variable_scope('discriminator', reuse=True):
        fake_output, fake_logit = Discriminator(fake_input)

    # 获得真实数据和伪数据的损失
    real_lost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit)*(1-args.smooth)))
    fake_lost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit)))
    all_d_lost = tf.add(fake_lost, real_lost)
    # 计算生成起生成的伪数据误差
    g_lost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit)*(1-args.smooth)))
    # 获取权重
    train_w = tf.trainable_variables()
    dis_weights = [var for var in train_w if var.name.startswith('discriminator')]
    gen_weights = [var for var in train_w if var.name.startswith('generator')]

    # 优化
    real_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(all_d_lost, var_list=dis_weights)
    fake_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(g_lost, var_list=gen_weights)

    # 全部变量的初始化
    init_all_v = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init_all_v)
    saver = load_model(sess)
    mnist = read_data_sets(args.data_dir)
    print('Saving images to: %s' %args.data_dir)
    fig_D_lost = []
    fig_G_lost = []
    for episode in range(args.episodes):
        for i in range(args.d_step):
            batch_x, _ = mnist.train.next_batch(args.batch_size)
            batch_x = batch_x.reshape(args.batch_size, 28, 28, 1)
            batch_x = batch_x*2 -1
            noise_x = np.random.uniform(-1 , 1, size=[args.batch_size, 28, 28, 1]).astype(np.float32)
            D_lost, R_lost, F_lost, _ = sess.run([all_d_lost, real_lost, fake_lost, real_optimizer], {real_input: batch_x, noise_input: noise_x})
            fig_D_lost.append(D_lost)

        for i in range(args.g_step):
            G_lost, _ = sess.run([g_lost, fake_optimizer], {noise_input: noise_x})
            fig_G_lost.append(G_lost)

        if episode % args.print_every == 0:
            print('Episode is: {0:d} Discriminator_Lost: {1:.3f} Generator_Lost: {2:.3f} Real_lost:{3:.3f} Fake_lost:{4:.3f}'.format(
                    episode, D_lost, G_lost, R_lost, F_lost))
        if episode % args.plot_every == 0 and episode != 0:
            plt.figure(figsize=(6*1.1618, 6))
            plt.plot(range(len(fig_D_lost)), fig_D_lost, color='red', label='discriminator')
            plt.plot(range(len(fig_G_lost)), fig_G_lost, color='blue', label='generator')
            plt.legend()
            plt.xlabel('iteration times')
            plt.ylabel('lost')
            plt.show()
        if episode % args.saveI_every == 0:
            file_name = os.path.join(args.result_dir, 'episode_%d_fake_imgae.jpg' % (episode))
            noise_x = np.random.uniform(-1, 1, [1, 28, 28, 1]).astype(np.float32)
            image_x = sess.run(fake_input, {noise_input:noise_x})
            # image_x = (image_x > 0.5).astype(np.float32)
            # image_x = (image_x < 0.5).astype(np.float32)
            imsave(file_name, image_x[0, :, :, 0])
        if episode % args.saveM_every == 0 and episode != 0:
            save_model(saver, sess, episode)
        if episode % args.saveW_every == 0 and episode > 0:
            nx = ny = 100
            canvas = np.empty((28*nx, 28*ny))
            for i in range(nx):
                for j in range(ny):
                    noise_x = np.random.uniform(-1, 1, [1, 28, 28, 1]).astype(np.float32)
                    image_fraction = sess.run(fake_input, {noise_input: noise_x})
                    image_fraction = np.reshape(image_fraction, [28, 28])
                    canvas[(i*28):(i+1)*28, (j*28):(j+1)*28] = image_fraction
            file_name = os.path.join(args.result_dir, 'episode_%d_whole_imgae.jpg' % (episode))
            imsave(file_name, canvas)
def save_model(saver, sess, step):
    saver.save(sess, os.path.join(args.network_dir, "gan_network"), global_step=step)

def load_model(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(args.network_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old weights!")

    return saver

def main(_):
    if tf.gfile.Exists(args.result_dir):
        tf.gfile.DeleteRecursively(args.result_dir)
    tf.gfile.MakeDirs(args.result_dir)
    train()
if __name__ == "__main__":
    tf.app.run()






