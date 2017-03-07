from __future__ import print_function

import math

import tensorflow as tf

from classifier.model import BaseModel


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                          'Global weight decay')


# model class
class MtcnnOnet(BaseModel):
  def inference(self, images, is_train):
    with tf.name_scope('model'):
      in_shape = images.get_shape().as_list()

      h = _conv2d(images, k_h=3, k_w=3, n_ch=32, s_h=2, s_w=2,
                  is_train=is_train, scope='conv_0', padding='VALID')
      h = tf.nn.relu(h)

      h = _conv2d(h, k_h=3, k_w=3, n_ch=64, s_h=2, s_w=2,
                  is_train=is_train, scope='conv_1', padding='VALID')
      h = tf.nn.relu(h)

      h = _conv2d(h, k_h=3, k_w=3, n_ch=64, s_h=2, s_w=2,
                  is_train=is_train, scope='conv_2', padding='VALID')
      h = tf.nn.relu(h)

      h = _conv2d(h, k_h=3, k_w=3, n_ch=128, s_h=2, s_w=2,
                  is_train=is_train, scope='conv_3', padding='VALID')
      h = tf.nn.relu(h)

      h = _conv2d(h, k_h=1, k_w=1, n_ch=256, s_h=1, s_w=1,
                  is_train=is_train, scope='conv_4', padding='VALID')
      h = tf.nn.relu(h)

      h = global_avg_pool(h, scope='global_pool')
      h = _conv2d(h, k_h=1, k_w=1, n_ch=10, s_h=1, s_w=1,
                  is_train=is_train, scope='fc')
      h = tf.squeeze(h, axis=[1, 2])
      return h


  def loss(self, gt_labels, labels):
    with tf.name_scope('loss') as scope:
      #ssd_loss = tf.reduce_mean(tf.nn.l2_loss(gt_labels - labels), name='landmarks_loss')
      ssd_loss = tf.reduce_mean(tf.square(gt_labels - labels), name='landmarks_loss')
      tf.add_to_collection('losses', ssd_loss)
      total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
      #tf.summary.scalar(scope + 'landmarks_loss', ssd_loss)
      tf.summary.scalar(scope + 'total_loss', total_loss)
      return total_loss


# utility functions
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def _variable_with_weight_decay(name, shape, wd):
  assert (len(shape) == 4)
  k_h, k_w, in_ch, out_ch = shape
  stddev = math.sqrt(2) * tf.sqrt(2. / ((out_ch + in_ch) * k_h * k_w))

  var = _variable_on_cpu(
    name,
    shape,
    tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='l2_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope, padding='SAME'):
  with tf.variable_scope(scope):
    in_ch = x.get_shape().as_list()[3]
    kernel = _variable_with_weight_decay('weights',
                                         shape=[k_h, k_w, in_ch, n_ch],
                                         wd=FLAGS.weight_decay)
    conv = tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding=padding)
    biases = _variable_on_cpu('biases', [n_ch], tf.constant_initializer(0.0))
    h = tf.nn.bias_add(conv, biases, name=scope + '_res')
    return h


def global_avg_pool(x, scope):
  ksize = [1, x.get_shape()[1], x.get_shape()[2], 1]
  return tf.nn.avg_pool(x, ksize=ksize, strides=[1, 1, 1, 1],
                        padding='VALID', name=scope)