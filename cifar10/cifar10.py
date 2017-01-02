# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import math

from six.moves import urllib
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10_input
import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def _scalar_summary(x, name=''):
  if name != '':
      name = '/' + name
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name) + name
  tf.scalar_summary(tensor_name, x)


def _histogram_summary(x, name=''):
  if name != '':
      name = '/' + name
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name) + name
  tf.histogram_summary(tensor_name, x)


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd, stddev=0.05):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  if FLAGS.glorot_init:
      k_h, k_w, in_ch, out_ch = shape
      stddev = math.sqrt(2) * tf.sqrt(2. / ((out_ch + in_ch) * k_h * k_w))

  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _sparse_variable_with_l1_loss(name, shape):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  k_h, k_w, in_ch, out_ch = shape

  # glorot initialization http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  # principle is taken from here https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py#L160 (relu case)
  stddev = math.sqrt(2) * tf.sqrt(2. / ((out_ch + in_ch) * k_h * k_w))
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  # from the paper https://arxiv.org/pdf/1611.06473v1.pdf (4.1)
  eps = FLAGS.c / stddev
  if FLAGS.mode == 'train':
      global_step = tf.contrib.framework.get_or_create_global_step()
      #if global_step >= FLAGS.lcnn_alpha_start_iter:
      alpha = tf.train.exponential_decay(FLAGS.alpha * eps, global_step,
                    decay_steps=FLAGS.lcnn_alpha_decay_step,
                    decay_rate=FLAGS.lcnn_alpha_decay, staircase=True)
      #else:
      #    alpha = FLAGS.alpha * eps
  elif FLAGS.mode == 'eval':
      alpha = 0
  else:
      raise Exception('Unknown mode {}'.format(FLAGS.mode))

  _scalar_summary(tf.identity(stddev, name='stddev'))
  _scalar_summary(tf.identity(eps, name='eps'))
  _scalar_summary(tf.identity(alpha, name='alpha'))

  var = var * tf.cast(tf.abs(var) > eps, tf.float32)
  sparsity = tf.nn.zero_fraction(var, name='weights_sparsity')
  _scalar_summary(sparsity)
  tf.add_to_collection('sparsities', sparsity)

  l1_loss = tf.mul(tf.reduce_sum(tf.abs(var)), alpha, name='l1_loss')
  tf.add_to_collection('losses', l1_loss)

  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference_small(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def inference_resnet(images, is_train):
    def conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope_name):
        with tf.variable_scope(scope_name):
            in_ch = x.get_shape().as_list()[3]
            kernel = _variable_with_weight_decay('weights',
                        shape=[k_h, k_w, in_ch, n_ch],
                        stddev=5e-2,
                        wd=FLAGS.weight_decay)
            conv = tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [n_ch], tf.constant_initializer(0.0))
            h = tf.nn.bias_add(conv, biases, name=scope_name+'_res')
            _activation_summary(h)
            return h

    def resnet_block(x, k_h, k_w, n_ch, downsample, is_train, scope_name):
        with tf.variable_scope(scope_name):
            in_depth = x.get_shape().as_list()[3]
            if downsample:
                s_h, s_w = 2, 2
                filter_ = [1,2,2,1]
                inpt = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')
            else:
                s_h, s_w = 1, 1
                inpt = x

            if in_depth != n_ch: # different number of channels
                inpt = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, n_ch - in_depth]])

            h = conv2d(x, k_h, k_w, n_ch, 1, 1,
                is_train=is_train, scope_name='conv1')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_1')
            h = tf.nn.relu(h)

            h = conv2d(h, k_h, k_w, n_ch, s_h, s_w,
                is_train=is_train, scope_name='conv2')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_2')
            h = tf.nn.relu(h + inpt)
            return h

    # taken from: https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua
    def resnet_block2(x, k_h, k_w, n_ch, downsample, is_train, scope_name):
        with tf.variable_scope(scope_name):
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
                is_train=is_train, scope_name='conv1')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_1')
            h = tf.nn.relu(h)

            h = conv2d(h, k_h, k_w, n_ch, 1, 1,
                is_train=is_train, scope_name='conv2')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_2')
            h = tf.nn.relu(h + inpt)
            return h

    n_blocks = 3
    n_f = 16
    in_shape = images.get_shape().as_list()

    if FLAGS.first_conv_3x3:
        h = conv2d(images, k_h=3, k_w=3, n_ch=n_f, s_h=1, s_w=1,
            is_train=is_train, scope_name='conv_init')
    else:
        h = conv2d(images, k_h=3, k_w=1, n_ch=n_f, s_h=1, s_w=1,
            is_train=is_train, scope_name='conv_init')

    h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
       trainable=True, is_training=is_train, reuse=False, scope='bn')
    h = tf.nn.relu(h)

    if FLAGS.resnet_block_type == 0:
      resnet_block = resnet_block
    elif FLAGS.resnet_block_type == 1:
      resnet_block = resnet_block2
    else:
      raise Exception('Unknown exception type {}'.format(FLAGS.resnet_block_type))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f, is_train=is_train,
            downsample=downsample, scope_name='resnet_1_{}'.format(i))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f*2, is_train=is_train,
            downsample=downsample, scope_name='resnet_2_{}'.format(i))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f*4, is_train=is_train,
            downsample=downsample, scope_name='resnet_3_{}'.format(i))

    h = tf.nn.avg_pool(h, ksize=[1,in_shape[1]/8,in_shape[2]/8,1],
                       strides=[1,1,1,1], padding='VALID')
    h = conv2d(h, k_h=1, k_w=1, n_ch=NUM_CLASSES, s_h=1, s_w=1,
               is_train=is_train, scope_name='fc')
    h = tf.squeeze(h)
    _activation_summary(h)
    return h


def inference_resnet_lcnn_hybrid(images, is_train):
    def conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope_name):
        with tf.variable_scope(scope_name):
            #in_ch = x.get_shape().as_list()[3]
            #kernel = _variable_with_weight_decay('weights',
            #            shape=[k_h, k_w, in_ch, n_ch],
            #            stddev=5e-2,
            #            wd=FLAGS.weight_decay)
            #conv = tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding='SAME')
            #biases = _variable_on_cpu('biases', [n_ch], tf.constant_initializer(0.0))
            #h = tf.nn.bias_add(conv, biases, name=scope_name+'_res')
            #_activation_summary(h)
            out_ch_1 = n_ch
            in_ch1 = x.get_shape().as_list()[3]

            kernel1 = _variable_with_weight_decay('weights1',
                        shape=[1, 1, in_ch1, out_ch_1],
                        stddev=5e-2,
                        wd=FLAGS.weight_decay)
            conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME', name='conv1x1')
            biases1 = _variable_on_cpu('biases1', [out_ch_1], tf.constant_initializer(0.0))
            h = tf.nn.bias_add(conv1, biases1, name=scope_name+'_res_0')

            if FLAGS.batch_norm_after_conv_1x1:
                h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                    trainable=True, is_training=is_train, reuse=False, scope='bn_0')

            in_ch2 = h.get_shape().as_list()[3]
            kernel2 = _variable_with_weight_decay('weights2',
                        shape=[k_h, k_w, in_ch2, n_ch],
                        stddev=5e-2,
                        wd=FLAGS.weight_decay)
            conv2 = tf.nn.conv2d(h, kernel2, [1, s_h, s_w, 1],
                                 padding='SAME', name='conv2')
            biases2 = _variable_on_cpu('biases2', [n_ch], tf.constant_initializer(0.0))
            h = tf.nn.bias_add(conv2, biases2, name=scope_name+'_res_1')
            return h

    def resnet_block(x, k_h, k_w, n_ch, downsample, is_train, scope_name):
        with tf.variable_scope(scope_name):
            in_depth = x.get_shape().as_list()[3]
            if downsample:
                s_h, s_w = 2, 2
                filter_ = [1,2,2,1]
                inpt = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')
            else:
                s_h, s_w = 1, 1
                inpt = x

            if in_depth != n_ch: # different number of channels
                inpt = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, n_ch - in_depth]])

            h = conv2d(x, k_h, k_w, n_ch, 1, 1,
                is_train=is_train, scope_name='conv1')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_1')
            h = tf.nn.relu(h)

            h = conv2d(h, k_h, k_w, n_ch, s_h, s_w,
                is_train=is_train, scope_name='conv2')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_2')
            h = tf.nn.relu(h + inpt)
            return h

    n_blocks = 3
    n_f = 16
    in_shape = images.get_shape().as_list()

    h = conv2d(images, k_h=3, k_w=3, n_ch=n_f, s_h=1, s_w=1,
        is_train=is_train, scope_name='conv_init')
    h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
        trainable=True, is_training=is_train, reuse=False, scope='bn')
    h = tf.nn.relu(h)

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f, is_train=is_train,
            downsample=downsample, scope_name='resnet_1_{}'.format(i))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f*2, is_train=is_train,
            downsample=downsample, scope_name='resnet_2_{}'.format(i))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = resnet_block(h, k_h=3, k_w=3, n_ch=n_f*4, is_train=is_train,
            downsample=downsample, scope_name='resnet_3_{}'.format(i))

    h = tf.nn.avg_pool(h, ksize=[1,in_shape[1]/8,in_shape[1]/8,1],
                       strides=[1,1,1,1], padding='VALID')
    h = conv2d(h, k_h=1, k_w=1, n_ch=NUM_CLASSES, s_h=1, s_w=1,
               is_train=is_train, scope_name='fc')
    h = tf.squeeze(h)
    _activation_summary(h)
    return h


def inference_lcnn_resnet(images, is_train):
    def sparse_lcnn_conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope_name):
        with tf.variable_scope(scope_name):
            out_ch_1 = n_ch / FLAGS.channels_reduction_ratio
            in_ch1 = x.get_shape().as_list()[3]

            kernel1 = _variable_with_weight_decay('weights1',
                        shape=[1, 1, in_ch1, out_ch_1],
                        stddev=5e-2,
                        wd=0)
            conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME', name='conv1x1')
            biases1 = _variable_on_cpu('biases1', [out_ch_1], tf.constant_initializer(0.0))
            h = tf.nn.bias_add(conv1, biases1, name=scope_name+'_res_0')

            if FLAGS.batch_norm_after_conv_1x1:
                h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                    trainable=True, is_training=is_train, reuse=False, scope='bn_0')

            in_ch2 = h.get_shape().as_list()[3]
            kernel2 = _sparse_variable_with_l1_loss('weights2',
                        shape=[k_h, k_w, in_ch2, n_ch])
            conv2 = tf.nn.conv2d(h, kernel2, [1, s_h, s_w, 1], padding='SAME', name='conv_sparse')
            biases2 = _variable_on_cpu('biases2', [n_ch], tf.constant_initializer(0.0))
            h = tf.nn.bias_add(conv2, biases2, name=scope_name+'_res_1')
            return h

    #def lcnn_conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope_name):
    #    with tf.variable_scope(scope_name):
    #        ch_scale = 4
    #        n_ch_1 = n_ch / ch_scale
    #        kernel1 = _variable_with_weight_decay('weights1',
    #                    shape=[1, 1, x.get_shape().as_list()[3], n_ch_1],
    #                    stddev=5e-2,
    #                    wd=0.0001)
    #        conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
    #        biases1 = _variable_on_cpu('biases1', [n_ch_1], tf.constant_initializer(0.0))
    #        h = tf.nn.bias_add(conv1, biases1)
    #
    #        kernel2 = _variable_with_weight_decay('weights2',
    #                    shape=[k_h, k_w, h.get_shape().as_list()[3], n_ch],
    #                    stddev=5e-2,
    #                    wd=0.0001)
    #        conv2 = tf.nn.conv2d(h, kernel2, [1, s_h, s_w, 1], padding='SAME')
    #        biases2 = _variable_on_cpu('biases2', [n_ch], tf.constant_initializer(0.0))
    #        h = tf.nn.bias_add(conv2, biases2, name=scope_name+'_res')
    #        _activation_summary(h)
    #        return h

    def conv2d(x, k_h, k_w, n_ch, s_h, s_w, is_train, scope_name):
        with tf.variable_scope(scope_name):
            kernel = _variable_with_weight_decay('weights',
                        shape=[k_h, k_w, x.get_shape().as_list()[3], n_ch],
                        stddev=5e-2,
                        wd=FLAGS.weight_decay)
            conv = tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [n_ch], tf.constant_initializer(0.0))
            h = tf.nn.bias_add(conv, biases, name=scope_name+'_res')
            return h

    def lcnn_resnet_block(x, k_h, k_w, n_ch, downsample, is_train, scope_name):
        with tf.variable_scope(scope_name):
            in_depth = x.get_shape().as_list()[3]
            if downsample:
                s_h, s_w = 2, 2
                filter_ = [1,2,2,1]
                inpt = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')
            else:
                s_h, s_w = 1, 1
                inpt = x

            if in_depth != n_ch: # different number of channels
                inpt = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, n_ch - in_depth]])

            h = sparse_lcnn_conv2d(x, k_h, k_w, n_ch, 1, 1,
                is_train=is_train, scope_name='conv1')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_1')
            h = tf.nn.relu(h)

            h = sparse_lcnn_conv2d(h, k_h, k_w, n_ch, s_h, s_w,
                is_train=is_train, scope_name='conv2')
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
                trainable=True, is_training=is_train, reuse=False, scope='bn_2')
            h = tf.nn.relu(h + inpt)
            return h

    #def resnet_block(x, k_h, k_w, n_ch, downsample, is_train, scope_name):
    #    with tf.variable_scope(scope_name):
    #        in_depth = x.get_shape().as_list()[3]
    #        if downsample:
    #            s_h, s_w = 2, 2
    #            filter_ = [1,2,2,1]
    #            inpt = tf.nn.max_pool(x, ksize=filter_, strides=filter_, padding='SAME')
    #        else:
    #            s_h, s_w = 1, 1
    #            inpt = x
    #
    #        if in_depth != n_ch: # different number of channels
    #            inpt = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, n_ch - in_depth]])
    #
    #        h = conv2d(x, k_h, k_w, n_ch, 1, 1,
    #            is_train=is_train, scope_name='conv1')
    #        h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
    #            trainable=True, is_training=is_train, reuse=False, scope='bn_1')
    #        h = tf.nn.relu(h)
    #
    #        h = conv2d(h, k_h, k_w, n_ch, s_h, s_w,
    #            is_train=is_train, scope_name='conv2')
    #        h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
    #            trainable=True, is_training=is_train, reuse=False, scope='bn_2')
    #        h = tf.nn.relu(h + inpt)
    #        return h

    n_blocks = 3
    n_f = 16
    in_shape = images.get_shape().as_list()

    h = sparse_lcnn_conv2d(images, k_h=3, k_w=3, n_ch=n_f, s_h=1, s_w=1,
        is_train=is_train, scope_name='conv_init')
    h = tf.contrib.layers.batch_norm(inputs=h, decay=0.95,
        trainable=True, is_training=is_train, reuse=False, scope='bn')
    h = tf.nn.relu(h)

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = lcnn_resnet_block(h, k_h=3, k_w=3, n_ch=n_f, is_train=is_train,
            downsample=downsample, scope_name='resnet_1_{}'.format(i))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = lcnn_resnet_block(h, k_h=3, k_w=3, n_ch=n_f*2, is_train=is_train,
            downsample=downsample, scope_name='resnet_2_{}'.format(i))

    for i in xrange(0, n_blocks):
        downsample = (i == n_blocks - 1)
        h = lcnn_resnet_block(h, k_h=3, k_w=3, n_ch=n_f*4, is_train=is_train,
            downsample=downsample, scope_name='resnet_3_{}'.format(i))

    h = tf.nn.avg_pool(h, ksize=[1,in_shape[1]/8,in_shape[1]/8,1],
                       strides=[1,1,1,1], padding='VALID')
    h = conv2d(h, k_h=1, k_w=1, n_ch=NUM_CLASSES, s_h=1, s_w=1,
               is_train=is_train, scope_name='fc')
    h = tf.squeeze(h)

    # output average sparsity
    avg_sparsity = tf.reduce_mean(tf.get_collection('sparsities'))
    tf.scalar_summary('average_sparsity', avg_sparsity)

    return h


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  decay_steps = FLAGS.num_updates_per_decay

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(FLAGS.initial_lr,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
