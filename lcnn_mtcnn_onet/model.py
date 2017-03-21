from __future__ import print_function

import math

import tensorflow as tf

from classifier.model import BaseModel
import classifier.tf_utils as tf_utils


tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                          'Global weight decay')
tf.app.flags.DEFINE_float('batch_norm_decay', 0.9,
                          'Global weight decay')
tf.app.flags.DEFINE_integer('num_base_filters', 32,
                          'Base for the filters (multiplied by coeff afterwards)')
tf.app.flags.DEFINE_float('face_ce_loss_coeff', 0.01,
                          'Coefficient to multiply face loss to')
tf.app.flags.DEFINE_float('bbox_loss_coeff', 0.1,
                          'Coefficient to multiply face loss to')
tf.app.flags.DEFINE_float('landmarks_loss_coeff', 3.,
                          'Coefficient to multiply face loss to')
FLAGS = tf.app.flags.FLAGS


# model class
class MtcnnOnet(BaseModel):
  def inference(self, images, is_train):
    with tf.name_scope('model'):
      in_shape = images.get_shape().as_list()

      n_f = FLAGS.num_base_filters
      h = _conv2d(images, k_h=3, k_w=3, n_ch=n_f, s_h=1, s_w=1,
                  is_train=is_train, scope='conv0', padding='SAME')
      h = tf.contrib.layers.batch_norm(inputs=h, decay=FLAGS.batch_norm_decay,
                                       trainable=True, is_training=is_train, reuse=False, scope='bn1')
      #h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID',
      #                   name='pool1')
      # for compatability with caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp#L90
      h = tf.nn.max_pool(h, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID',
                         name='pool1')
      h = tf.nn.relu(h)
      print('conv0', h)

      h = _conv2d(h, k_h=3, k_w=3, n_ch=n_f*2, s_h=1, s_w=1,
                  is_train=is_train, scope='conv2', padding='SAME')
      h = tf.contrib.layers.batch_norm(inputs=h, decay=FLAGS.batch_norm_decay,
                                       trainable=True, is_training=is_train, reuse=False, scope='bn2')
      #h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID',
      #                   name='pool3')
      h = tf.nn.max_pool(h, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID',
                         name='pool3')
      h = tf.nn.relu(h)
      print('conv2', h)

      h = _conv2d(h, k_h=3, k_w=3, n_ch=n_f*2, s_h=1, s_w=1,
                  is_train=is_train, scope='conv4', padding='SAME')
      h = tf.contrib.layers.batch_norm(inputs=h, decay=FLAGS.batch_norm_decay,
                                       trainable=True, is_training=is_train, reuse=False, scope='bn3')
      h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID',
                         name='pool5')
      h = tf.nn.relu(h)
      print('conv4', h)

      h = _conv2d(h, k_h=3, k_w=3, n_ch=n_f*4, s_h=1, s_w=1,
                  is_train=is_train, scope='conv6', padding='VALID')
      h = tf.contrib.layers.batch_norm(inputs=h, decay=FLAGS.batch_norm_decay,
                                       trainable=True, is_training=is_train, reuse=False, scope='bn4')
      h = tf.nn.relu(h)
      print('conv6', h)

      h = _conv2d(h, k_h=1, k_w=1, n_ch=n_f*8, s_h=1, s_w=1,
                  is_train=is_train, scope='conv7', padding='VALID')
      h = tf.contrib.layers.batch_norm(inputs=h, decay=FLAGS.batch_norm_decay,
                                       trainable=True, is_training=is_train, reuse=False, scope='bn5')
      h = tf.nn.relu(h)
      print('conv7', h)

      # try to remove this one?
      h = global_avg_pool(h, scope='global_pool')
      h1 = tf.squeeze(_conv2d(h, k_h=1, k_w=1, n_ch=2, s_h=1, s_w=1,
                  is_train=is_train, scope='conv_face'), axis=[1,2])
      h2 = tf.squeeze(_conv2d(h, k_h=1, k_w=1, n_ch=4, s_h=1, s_w=1,
                   is_train=is_train, scope='conv_bbox'), axis=[1,2])
      h3 = tf.squeeze(_conv2d(h, k_h=1, k_w=1, n_ch=10, s_h=1, s_w=1,
                   is_train=is_train, scope='conv_landmarks'), axis=[1,2])
      return h1, h2, h3, tf.nn.softmax(h1)


  # batch is composed of:
  # faces part
  # partfaces part
  # not_faces part
  # landmarks part
  # we split it accordingly
  def loss(self, gt_labels, preds):
    with tf.name_scope('loss') as scope:
      gt_type = tf.cast(gt_labels[:, 0], dtype=tf.int32)
      n_faces = tf_utils.count(gt_type, 0)
      n_partfaces = tf_utils.count(gt_type, 1)
      n_not_faces = tf_utils.count(gt_type, 2)
      n_landmarks = tf_utils.count(gt_type, 3)

      zero = tf_utils.zero_const()
      minus_one = tf_utils.minus_one_const()

      faces_range = [(zero, n_faces), (n_faces + n_partfaces,  n_not_faces)]
      bbox_range = (zero, n_faces + n_partfaces)
      landmarks_range = (n_faces + n_partfaces + n_not_faces, n_landmarks)

      preds_face = preds[0]
      preds_bbox = preds[1]
      preds_landmarks = preds[2]

      gt_face = tf.cast(gt_labels[:, 1], dtype=tf.int32)
      gt_bbox = gt_labels[:, 2:6]
      gt_landmarks = gt_labels[:, 6:16]

      # cross-entropy
      gt_face_slice = tf.concat([tf.slice(gt_face, faces_range[0][0], faces_range[0][1]),
                                 tf.slice(gt_face, faces_range[1][0], faces_range[1][1])], axis=0)

      pred_face_slice1 = tf.slice(preds_face, tf_utils.stack_and_squeeze([faces_range[0][0], zero]),
                                  tf_utils.stack_and_squeeze([faces_range[0][1], minus_one]))
      pred_face_slice2 = tf.slice(preds_face, tf_utils.stack_and_squeeze([faces_range[1][0], zero]),
                                  tf_utils.stack_and_squeeze([faces_range[1][1], minus_one]))
      pred_face_slice = tf.concat([pred_face_slice1, pred_face_slice2], axis=0)
      ce_face = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=gt_face_slice, logits=pred_face_slice))
      ce_face_loss = tf.multiply(FLAGS.face_ce_loss_coeff, ce_face, name='cross_entropy')

      # top-n face accuracy
      softmax_preds = tf.nn.softmax(pred_face_slice)
      _, index = tf.nn.top_k(softmax_preds)
      correct_preds_arr = tf.equal(tf.reshape(gt_face_slice, [-1]), tf.reshape(index, [-1]))
      top1_face_acc = tf.reduce_sum(tf.cast(correct_preds_arr, dtype=tf.float32)) / \
                      tf.cast(tf.shape(gt_face_slice)[0], dtype=tf.float32)

      # bbox
      begin = tf_utils.stack_and_squeeze([bbox_range[0], zero])
      size = tf_utils.stack_and_squeeze([bbox_range[1], minus_one])
      ssd_bbox = tf.reduce_mean(tf.square(tf.slice(gt_bbox, begin, size) - tf.slice(preds_bbox, begin, size)),)
      ssd_bbox_loss = tf.multiply(FLAGS.bbox_loss_coeff, ssd_bbox, name='bbox_loss')

      # landmarks
      begin = tf_utils.stack_and_squeeze([landmarks_range[0], zero])
      size = tf_utils.stack_and_squeeze([landmarks_range[1], minus_one])
      ssd_landmarks = tf.reduce_mean(tf.square(tf.slice(gt_landmarks, begin, size) -
                                               tf.slice(preds_landmarks, begin, size)))
      ssd_landmarks_loss = tf.multiply(FLAGS.landmarks_loss_coeff, ssd_landmarks, name='landmarks_loss')

      tf.add_to_collection('losses', ce_face_loss)
      tf.add_to_collection('losses', ssd_bbox_loss)
      tf.add_to_collection('losses', ssd_landmarks_loss)

      total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
      tf.summary.scalar(scope + 'cross_entropy', ce_face)
      tf.summary.scalar(scope + 'bbox_ssd', ssd_bbox)
      tf.summary.scalar(scope + 'landmarks_ssd', ssd_landmarks)
      tf.summary.scalar(scope + 'top1_accuracy', top1_face_acc)
      tf.summary.scalar(scope + 'total_loss', total_loss)

      return [total_loss, ce_face, top1_face_acc, ssd_bbox, ssd_landmarks]


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