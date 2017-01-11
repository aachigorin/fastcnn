from __future__ import print_function

import tensorflow as tf
import time
import datetime
import os
import re
import numpy as np

from reader import Cifar10Reader 
from model import Cifar10Resnet18


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('gpus', '0',
                           'Available gpus')
tf.app.flags.DEFINE_float('gpu_memory_ratio', 0.4,
                           'Ration of gpu memory to lock')


tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('optimizer', 'sgd_momentum',
                           """Optimizer of choice""")
tf.app.flags.DEFINE_float('initial_lr', 0.1,
                          """Learning rate to start with""")
tf.app.flags.DEFINE_integer('num_updates_per_decay', 30000,
                            """Number of updates after which we drop learning rate""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1,
                          """Decay value""")


def tower_loss(images, labels, model, is_train, scope):
  logits = model.inference(images, is_train=is_train)
  _, top1_loss = model.loss(logits, labels)
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  for l in losses:
    loss_name = re.sub('%tower_[0-9]*/', '', l.op.name)
    tf.summary.scalar(loss_name, l)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
  updates = tf.group(*update_ops) # For correct batch norm execution
  with tf.control_dependencies([updates, top1_loss]):
    total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, var in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  train_graph = tf.Graph()
  with train_graph.as_default() as g:
    with g.name_scope('train'):
      # readers
      with tf.name_scope('trainer_reader') as scope:
        train_reader = Cifar10Reader(data_dir=FLAGS.data_dir, batch_size=128,
                               part=Cifar10Reader.DatasetPart.train,
                               preprocessing=Cifar10Reader.Preprocessing.random_simple)
        images, labels = train_reader.get_batch()
        reader_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

      # optimizer
      global_step = tf.get_variable('global_step', [],
                      initializer=tf.constant_initializer(0),
                      trainable=False)

      lr = tf.train.exponential_decay(FLAGS.initial_lr,
                                      global_step,
                                      FLAGS.num_updates_per_decay,
                                      FLAGS.lr_decay_factor,
                                      staircase=True)

      if FLAGS.optimizer == 'sgd_momentum':
          opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
      elif FLAGS.optimizer == 'adam':
          opt = tf.train.AdamOptimizer(learning_rate=lr)
      else:
          raise Exception('Unknown optimizer type {}'.format(FLAGS.optimizer))

      tower_grads = []
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:{}'.format(i)):
          with tf.name_scope('t_{}'.format(i)) as scope:
            model = Cifar10Resnet18()
            loss = tower_loss(images, labels, model,
                              is_train=True, scope=scope)
            tf.get_variable_scope().reuse_variables()
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)

      grads = average_gradients(tower_grads)
      train_op = opt.apply_gradients(grads, global_step=global_step)

      # saver
      saver = tf.train.Saver(tf.global_variables())

      summaries.append(tf.summary.scalar('learning_rate', lr))
      summary_op = tf.summary.merge(summaries + reader_summaries)

      # creating a session
      config = tf.ConfigProto()
      config.gpu_options.per_process_gpu_memory_fraction=FLAGS.gpu_memory_ratio
      config.gpu_options.visible_device_list=FLAGS.gpus
      config.allow_soft_placement=True
      config.log_device_placement=FLAGS.log_device_placement
      config.gpu_options.allow_growth=True
      sess = tf.Session(config=config)
      sess.run(tf.global_variables_initializer())

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / FLAGS.num_gpus

          format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (step, loss_value,
                               examples_per_sec, sec_per_batch))

        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
