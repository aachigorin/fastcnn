from __future__ import print_function

import time
import os
import re

import numpy as np
import tensorflow as tf


__all__ = ['train']


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('loss_output_freq', 50,
                            """How often to print loss value to stdout.""")
tf.app.flags.DEFINE_integer('summary_write_freq', 1000,
                            """How often to write summary to disk.""")
tf.app.flags.DEFINE_integer('backup_freq', 1000,
                            """How often to save the model.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('gpus', '0',
                           'Available gpus')
tf.app.flags.DEFINE_float('gpu_memory_ratio', 0.8,
                           'Ration of gpu memory to lock')


tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 70000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('optimizer', 'sgd_momentum',
                           """Optimizer of choice""")
tf.app.flags.DEFINE_string('lr_schedule', '0:0.1,32000:0.01,48000:0.001',
                          """Learning rate to start with""")


def train(create_model, create_optimizer, create_reader):
  with tf.get_default_graph().name_scope('train'):
    with tf.name_scope('trainer_reader') as scope:
      reader = create_reader()
      images, labels = reader.get_batch()
      reader_sum = tf.get_collection(tf.GraphKeys.SUMMARIES, scope) 

    images_splits = tf.split(images, axis=0, num_or_size_splits=FLAGS.num_gpus)
    labels_splits = tf.split(labels, axis=0, num_or_size_splits=FLAGS.num_gpus)

    with tf.name_scope('optimizer') as scope:
      opt = create_optimizer()
      global_step = tf.train.get_global_step()
      optimizer_sum = tf.get_collection(tf.GraphKeys.SUMMARIES, scope) 

    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:{}'.format(i)):
        with tf.name_scope('t_{}'.format(i)) as scope:
          model = create_model()
          loss, top1_acc = _tower_loss(images_splits[i], labels_splits[i], model,
                             is_train=True, scope=scope)
          # keep summaries only from one of the towers
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
          grads = opt.compute_gradients(loss)
          tower_grads.append(grads)
          # why it does not work like this??
          assert(FLAGS.num_gpus == 1)
          #tf.get_variable_scope().reuse_variables()

          for grad, var in grads:
            if grad is not None:
              summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    grads = _average_gradients(tower_grads)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge(summaries + reader_sum + optimizer_sum)

    # creating a session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=FLAGS.gpu_memory_ratio
    config.gpu_options.visible_device_list=FLAGS.gpus
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    config.log_device_placement=FLAGS.log_device_placement
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # start the queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables())

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_val, top1_acc_val = sess.run([train_op, loss, top1_acc])
      #assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

      if step % FLAGS.loss_output_freq == 0:
        duration = time.time() - start_time
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('step %d, loss = %.2f, top1_ac = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (step, loss_val, top1_acc_val,
                             examples_per_sec, sec_per_batch))

      if step % FLAGS.summary_write_freq == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % FLAGS.backup_freq == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def _tower_loss(images, labels, model, is_train, scope):
  logits = model.inference(images, is_train=is_train)
  _, top1_acc = model.loss(logits, labels)
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  for l in losses:
    loss_name = re.sub('%tower_[0-9]*/', '', l.op.name)
    tf.summary.scalar(loss_name, l)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  updates = tf.group(*update_ops) # for correct batch norm execution
  with tf.control_dependencies([updates, top1_acc]):
    total_loss = tf.identity(total_loss)
  return total_loss, top1_acc


def _average_gradients(tower_grads):
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
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
