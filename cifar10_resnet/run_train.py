import tensorflow as tf

import fastcnn.classifier.trainer as trainer
from model import Cifar10Resnet18
from reader import Cifar10Reader


FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir) 

  def create_reader():
    return Cifar10Reader(data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size,
            part=Cifar10Reader.DatasetPart.train,
            preprocessing=Cifar10Reader.Preprocessing.random_simple)

  def create_model():
    return Cifar10Resnet18()

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

  trainer.train(create_model, create_optimizer, create_reader)


if __name__ == '__main__':
  tf.app.run()
