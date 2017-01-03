from __future__ import print_function

from abc import ABCMeta, abstractmethod
import math

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                           'Global weight decay')


class Model(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def inference(self, images, is_train):
    pass


  @abstractmethod
  def loss(self, logits, labels):
    pass


# utility functions
def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def variable_with_weight_decay(name, shape, wd):
    assert(len(shape) == 4)
    k_h, k_w, in_ch, out_ch = shape
    stddev = math.sqrt(2) * tf.sqrt(2. / ((out_ch + in_ch) * k_h * k_w))

    var = variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

    if wd is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='l2_loss')
      tf.add_to_collection('losses', weight_decay)

    return var


def conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope):
  with tf.variable_scope(scope):
    in_ch = x.get_shape().as_list()[3]
    kernel = variable_with_weight_decay('weights',
                shape=[k_h, k_w, in_ch, n_ch],
                wd=FLAGS.weight_decay)
    conv = tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding='SAME')
    biases = variable_on_cpu('biases', [n_ch], tf.constant_initializer(0.0))
    h = tf.nn.bias_add(conv, biases, name=scope+'_res')
    return h


# taken from: https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua
def resnet_block(x, k_h, k_w, n_ch, downsample, is_train, scope):
    with tf.variable_scope(scope):
        in_depth = x.get_shape().as_list()[3]
        if downsample:
            s_h, s_w = 2, 2
            filter_ = [1,2,2,1]
            inpt = tf.nn.avg_pool(x, ksize=filter_, strides=filter_, padding='SAME')
        else:
            s_h, s_w = 1, 1
            inpt = x

        if in_depth != n_ch: # different number of channels
            inpt = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, n_ch - in_depth]])

        h = conv2d(x, k_h, k_w, n_ch, s_h, s_w,
            is_train=is_train, scope='conv1')
        h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
            trainable=True, is_training=is_train, reuse=False, scope='bn_1')
        h = tf.nn.relu(h)

        h = conv2d(h, k_h, k_w, n_ch, 1, 1,
            is_train=is_train, scope='conv2')
        h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
            trainable=True, is_training=is_train, reuse=False, scope='bn_2')
        h = tf.nn.relu(h + inpt)
        return h


# model class
class Cifar10Resnet18(Model):
  NUM_CLASSES = 10

  def inference(self, images, is_train):
    with tf.name_scope('resnet18_model'):
      n_blocks = 3
      n_f = 16
      in_shape = images.get_shape().as_list()

      h = conv2d(images, k_h=3, k_w=3, n_ch=n_f, s_h=1, s_w=1,
          is_train=is_train, scope='conv_0')
      h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
         trainable=True, is_training=is_train, reuse=False, scope='bn')
      h = tf.nn.relu(h)

      for i in xrange(0, n_blocks):
          downsample = (i == n_blocks - 1)
          h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f, is_train=is_train,
              downsample=downsample, scope='resnet_1_{}'.format(i))

      for i in xrange(0, n_blocks):
          downsample = (i == n_blocks - 1)
          h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f*2, is_train=is_train,
              downsample=downsample, scope='resnet_2_{}'.format(i))

      for i in xrange(0, n_blocks):
          downsample = (i == n_blocks - 1)
          h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f*4, is_train=is_train,
              downsample=downsample, scope='resnet_3_{}'.format(i))

      h = tf.nn.avg_pool(h, ksize=[1,in_shape[1]/8,in_shape[2]/8,1],
                         strides=[1,1,1,1], padding='VALID')
      h = tf.reshape(h, [in_shape[0], -1])
      h = tf.contrib.layers.fully_connected(inputs=h, activation_fn=None,
            num_outputs=Cifar10Resnet18.NUM_CLASSES, scope='fc')
      h = tf.squeeze(h)
      return h


  def loss(self, logits, labels):
    with tf.name_scope('resnet18_loss'):
      labels = tf.cast(labels, tf.int64)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection('losses', cross_entropy_mean)

      top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))

      # The total loss is defined as the cross entropy loss plus all of the weight
      # decay terms (L2 loss).
      return tf.add_n(tf.get_collection('losses'), name='total_loss'), top_1
