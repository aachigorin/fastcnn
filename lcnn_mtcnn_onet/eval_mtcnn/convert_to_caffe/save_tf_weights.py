import tensorflow as tf
import os
import scipy.io
import numpy as np

from lcnn_mtcnn_onet.model import MtcnnOnet


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_weights_to', '/tmp/',
                           """Directory where to write mtcnn weight""")


def main(argv=None):  # pylint: disable=unused-argument
  tf.set_random_seed(2410)
  model = MtcnnOnet()
  img_size = 48
  images = tf.placeholder(tf.float32, shape=(1, img_size, img_size, 3))
  predictions = model.inference(images, is_train=False)
  #all_losses = model.loss(labels, predictions)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.1
  config.gpu_options.visible_device_list = '0'
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  sess.run(tf.global_variables_initializer())

  # restoring the model
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(os.path.abspath(FLAGS.train_dir))
  global_step = _get_global_step(ckpt)
  assert(global_step is not None)
  saver.restore(sess, ckpt.model_checkpoint_path)

  print('Trainable variables:')
  trainable_vars = tf.global_variables()
  for var in trainable_vars:
    print(var)

  conv0 = get_conv_params(trainable_vars, 'conv0')
  bn1 = get_bn_params(trainable_vars, 'bn1')
  conv0_merged = merge_batch_norm_after_conv(conv0, bn1)

  conv2 = get_conv_params(trainable_vars, 'conv2')
  bn2 = get_bn_params(trainable_vars, 'bn2')
  conv2_merged = merge_batch_norm_after_conv(conv2, bn2)

  conv4 = get_conv_params(trainable_vars, 'conv4')
  bn3 = get_bn_params(trainable_vars, 'bn3')
  conv4_merged = merge_batch_norm_after_conv(conv4, bn3)

  conv6 = get_conv_params(trainable_vars, 'conv6')
  bn4 = get_bn_params(trainable_vars, 'bn4')
  conv6_merged = merge_batch_norm_after_conv(conv6, bn4)

  conv7 = get_conv_params(trainable_vars, 'conv7')
  bn5 = get_bn_params(trainable_vars, 'bn5')
  conv7_merged = merge_batch_norm_after_conv(conv7, bn5)

  conv_face = get_conv_params(trainable_vars, 'conv_face')
  conv_bbox = get_conv_params(trainable_vars, 'conv_bbox')
  conv_landmarks = get_conv_params(trainable_vars, 'conv_landmarks')

  def save_conv_weights(weights, name):
    scipy.io.savemat('{}/{}_weights.mat'.format(FLAGS.save_weights_to, name),
                     {'x': weights['w'].tolist()})
    scipy.io.savemat('{}/{}_biases.mat'.format(FLAGS.save_weights_to, name),
                     {'x': weights['b'].tolist()})

  save_conv_weights(sess.run(conv0_merged), 'conv0')
  save_conv_weights(sess.run(conv2_merged), 'conv2')
  save_conv_weights(sess.run(conv4_merged), 'conv4')
  save_conv_weights(sess.run(conv6_merged), 'conv6')
  save_conv_weights(sess.run(conv7_merged), 'conv7')
  save_conv_weights(sess.run(conv_face), 'conv_face')
  save_conv_weights(sess.run(conv_bbox), 'conv_bbox')
  save_conv_weights(sess.run(conv_landmarks), 'conv_landmarks')

  test_mat_size = 48
  test_mat = np.ones((1, test_mat_size, test_mat_size, 3))
  for i in xrange(test_mat_size):
    for j in xrange(test_mat_size):
      for k in xrange(3):
        test_mat[0, i, j, k] = i + j + k

  def save_outputs(fd, layer, name):
    outputs = sess.run(layer, feed_dict=fd)
    scipy.io.savemat('{}/{}_outputs.mat'.format(FLAGS.save_weights_to, name),
                     {'x': outputs.tolist()})

  save_outputs({images: test_mat}, model.conv0, 'conv0')
  save_outputs({images : test_mat}, model.bn1, 'bn1')
  save_outputs({images: test_mat}, model.conv2, 'conv2')
  save_outputs({images: test_mat}, model.bn2, 'bn2')
  save_outputs({images: test_mat}, model.bn3, 'bn3')
  save_outputs({images: test_mat}, model.bn4, 'bn4')
  #save_outputs({images: test_mat}, model.avg_pool, 'avg_pool')
  save_outputs({images: test_mat}, model.face_softmax, 'face_softmax')
  save_outputs({images: test_mat}, model.bbox, 'bbox')
  save_outputs({images: test_mat}, model.landmarks, 'landmarks')


def _get_global_step(ckpt):
  if ckpt and ckpt.model_checkpoint_path:
    # assuming model_checkpoint_path looks something like:
    # /my-favorite-path/imagenet_train/model.ckpt-0,
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, global_step))
  else:
    print('No checkpoint file found')
    global_step = None
  return global_step


def get_conv_params(vars, prefix):
  w = None
  b = None
  for v in vars:
    if v.name.startswith(prefix + '/weights') and w is None:
      w = v
    if v.name.startswith(prefix + '/biases') and b is None:
      b = v
  assert(w is not None)
  assert(b is not None)
  return {'w': w, 'b': b}


def get_bn_params(vars, prefix):
  beta = None
  mean = None
  var = None
  for v in vars:
    if v.name.startswith(prefix + '/beta') and beta is None:
      beta = v
    if v.name.startswith(prefix + '/moving_mean') and mean is None:
      mean = v
    if v.name.startswith(prefix + '/moving_variance') and var is None:
      var = v
  assert (beta is not None)
  assert (mean is not None)
  assert (var is not None)
  return {'beta' : beta, 'mean' : mean, 'var' : var}


def merge_batch_norm_after_conv(conv, bn):
  eps = 0.001
  beta, bn_mean, bn_var = bn['beta'], bn['mean'], bn['var']
  w = conv['w'] / tf.sqrt(bn_var + eps)
  b = (conv['b'] - bn_mean) / tf.sqrt(bn_var + eps) + beta
  return {'w' : w, 'b' : b}


if __name__ == '__main__':
  tf.app.run()