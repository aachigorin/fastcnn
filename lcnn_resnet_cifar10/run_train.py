import tensorflow as tf

import classifier.trainer as trainer
from dataset.cifar10_reader import Cifar10Reader

from model import Cifar10LCNNResnet18


FLAGS = tf.app.flags.FLAGS


def create_reader():
  def random_simple_preprocessor(image):
    with tf.name_scope('random_simple_preprocess'):
      image = (image - Cifar10Reader.MEAN) / Cifar10Reader.STD
      image = tf.image.random_flip_left_right(image)
      image = tf.image.resize_image_with_crop_or_pad(image,
                        Cifar10Reader.WIDTH + 2 * 4, Cifar10Reader.HEIGHT + 2 * 4)
      image = tf.random_crop(image, [Cifar10Reader.WIDTH, Cifar10Reader.HEIGHT, 3])
    return image

  return Cifar10Reader(data_dir=FLAGS.data_dir,
          batch_size=FLAGS.batch_size,
          part=Cifar10Reader.DatasetPart.train,
          processor=random_simple_preprocessor)


def create_model():
  return Cifar10LCNNResnet18()


def create_optimizer():
  class OptimizerType(object):
    sgd_momentum = 'sgd_momentum'
    adam = 'adam'

  global_step = tf.get_variable('global_step', [],
                  initializer=tf.constant_initializer(0),
                  trainable=False,
                  dtype=tf.int32)

  boundaries = [int(x.split(':')[0]) for x in FLAGS.lr_schedule.split(',')][1:]
  values = [float(x.split(':')[1]) for x in FLAGS.lr_schedule.split(',')]
  lr = tf.train.piecewise_constant(global_step, boundaries, values)

  if FLAGS.optimizer == OptimizerType.sgd_momentum:
      opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
  elif FLAGS.optimizer == OptimizerType.adam:
      opt = tf.train.AdamOptimizer(lr)
  else:
      raise Exception('Unknown optimizer type {}'.format(FLAGS.optimizer))

  tf.summary.scalar('learning_rate', lr)
  return opt


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir) 

  trainer.train(create_model, create_optimizer, create_reader)


if __name__ == '__main__':
  tf.app.run()
