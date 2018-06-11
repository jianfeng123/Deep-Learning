import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import os
import numpy as np
from scipy.misc import imsave
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

flags = tf.app.flags
flags.DEFINE_string('data_dir','Dat/mnist','directory for mnist')
flags.DEFINE_string('fig_dir','Dat/fig','directory for figure')

flags.DEFINE_string('latent_dim', 100, 'Latent dimensionality of model')
flags.DEFINE_string('batch_size', 64, 'Minibatch size')
flags.DEFINE_string('n_samples', 1, 'Samples for saving')
flags.DEFINE_string('hidden_size', 400, 'Hidden size of model')
flags.DEFINE_string('n_episodes', 100000, 'number of episodes')

FLAGS = flags.FLAGS

w_init = tf.random_normal_initializer(stddev=0.01)
def Encoder(x, latent_dim, hidden_size):
    x = tf.contrib.layers.flatten(x)
    layer1 = tf.layers.dense(x, hidden_size, activation=tf.nn.relu, kernel_initializer=w_init)
    layer2 = tf.layers.dense(layer1, hidden_size, activation=tf.nn.relu, kernel_initializer=w_init)
    output = tf.layers.dense(layer2, 2*latent_dim, activation=None, kernel_initializer=w_init)
    mu = output[:, :latent_dim]
    sigma = tf.nn.softplus(output[:, latent_dim:])
    return mu, sigma

def Decoder(z, hidden_size):
    layer1 = tf.layers.dense(z, hidden_size, activation=tf.nn.relu, kernel_initializer=w_init)
    layer2 = tf.layers.dense(layer1, hidden_size, activation=tf.nn.relu, kernel_initializer=w_init)
    output = tf.layers.dense(layer2, 28*28, activation=None, kernel_initializer=w_init)
    bernoulli_logits = tf.reshape(output, [-1, 28, 28, 1])
    return bernoulli_logits

def network_train():
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    with tf.name_scope('variational'):
        q_mu, q_sigma = Encoder(x, latent_dim=FLAGS.latent_dim, hidden_size=FLAGS.hidden_size)
        q_z = distributions.Normal(loc=q_mu,scale=q_sigma)
        assert  q_z.reparameterization_type == distributions.FULLY_REPARAMETERIZED
    with tf.variable_scope('model'):
        p_xIz_logits = Decoder(q_z.sample(), hidden_size=FLAGS.hidden_size)
        p_xIz = distributions.Bernoulli(logits=p_xIz_logits)
        posterior_predictive_samples = p_xIz.sample()
    with tf.variable_scope('model', reuse=True):
        p_z = distributions.Normal(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                                   scale=np.ones(FLAGS.latent_dim,dtype=np.float32))
        p_z_sample = p_z.sample(FLAGS.n_samples)
        p_xIz_logits = Decoder(p_z_sample, hidden_size=FLAGS.hidden_size)
        prior_predictive = distributions.Bernoulli(logits=p_xIz_logits)
        prior_predictive_samples = prior_predictive.sample()
    with tf.variable_scope('model', reuse=True):
        z_input = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
        p_xIz_logits = Decoder(z_input, hidden_size=FLAGS.hidden_size)
        prior_predictive_inp = distributions.Bernoulli(logits=p_xIz_logits)
        prior_predictive_inp_sample = prior_predictive_inp.sample()
    kl = tf.reduce_sum(distributions.kl(q_z, p_z),1)
    e_log_likelihood = tf.reduce_sum(p_xIz.log_prob(x), [1, 2, 3])
    elbo = tf.reduce_sum(e_log_likelihood - kl, 0)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(-elbo)
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)
    mnist = read_data_sets(FLAGS.data_dir)
    print('Saving images to: %s' % FLAGS.fig_dir)
    plot_elbo = []
    for i in range(FLAGS.n_episodes):
        batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
        batch_x = batch_x.reshape(FLAGS.batch_size, 28, 28, 1)
        batch_x = (batch_x>0.5).astype(np.float32)
        sess.run(optimizer, {x: batch_x})
        batch_elbo = sess.run(elbo, {x: batch_x})
        plot_elbo.append(batch_elbo/float(FLAGS.batch_size))
        if i % 1000 == 0:
            batch_elbo = sess.run(elbo, {x:batch_x})
            print('Episode: {0:d} ELBO: {1: .3f}'.format(i, batch_elbo/FLAGS.batch_size))
            batch_posterior_predictive_samples, batch_prior_predictive_samples = sess.run(
                [posterior_predictive_samples, prior_predictive_samples], {x:batch_x}
            )
            for k in range(FLAGS.n_samples):
                f_name = os.path.join(FLAGS.fig_dir, 'episode_%d_data_%d.jpg' % (i, k))
                imsave(f_name, batch_x[k, :, :, 0])
                f_name = os.path.join(FLAGS.fig_dir, 'episode_%d_posterior_%d.jpg' % (i, k))
                imsave(f_name, batch_posterior_predictive_samples[k, :, :, 0])
                f_name = os.path.join(FLAGS.fig_dir, 'episode_%d_prior_%d.jpg' % (i, k))
                imsave(f_name, batch_prior_predictive_samples[k, :, :, 0])
    plt.plot(range(len(plot_elbo)), plot_elbo)
    plt.show()

def main(_):
    if tf.gfile.Exists(FLAGS.fig_dir):
        tf.gfile.DeleteRecursively(FLAGS.fig_dir)
    tf.gfile.MakeDirs(FLAGS.fig_dir)
    network_train()
if __name__ == "__main__":
    tf.app.run()






