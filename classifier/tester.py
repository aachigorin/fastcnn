import os
import time

import numpy as np

import tensorflow as tf

__all__ = ["test"]


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_examples', '10000',
                           """.""")
tf.app.flags.DEFINE_string('dataset_part', 'test',
                           """.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', '300',
                           """.""")
tf.app.flags.DEFINE_bool('run_once', False,
                           """.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                           """.""")
tf.app.flags.DEFINE_string('gpus', '0',
                           'Available gpus')


def test(model, reader):
    with tf.name_scope('tester_reader') as scope:
      with tf.device('/cpu:0'):
        reader = reader()
        images, labels = reader.get_batch()
        reader_summaries = tf.summary.merge(
          tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

    model = model()
    predictions = model.inference(images, is_train=False)
    all_losses = model.loss(labels, predictions)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                              tf.get_default_graph())

    # creating Session only once in order to not initialize reader several times (seems to be a memory leak somewhere
    # because of this
    while True:
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = FLAGS.gpus
      #sess = tf.Session(config=config)

      with tf.Session(config=config) as sess:
        reader.init(sess)

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(os.path.abspath(FLAGS.train_dir))
        global_step = _get_global_step(ckpt)
        if global_step == None:
          time.sleep(FLAGS.eval_interval_secs)
          continue
        saver.restore(sess, ckpt.model_checkpoint_path)

        num_iters = FLAGS.num_examples // FLAGS.batch_size
        sum_all_losses = np.zeros((len(all_losses)))
        #count_top1 = 0.
        for i in xrange(num_iters):
          all_losses_val, reader_summaries_str = sess.run([all_losses, reader_summaries])
          sum_all_losses += all_losses_val
        avg_all_losses = sum_all_losses / num_iters

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        # adding summaries
        summary_writer.add_summary(reader_summaries_str, global_step)

        summary = tf.Summary()
        for i in xrange(len(avg_all_losses)):
          summary.value.add(tag='tester/avg_loss_{}'.format(i), simple_value=avg_all_losses[i])
          summary_writer.add_summary(summary, global_step)

      print('{} results. all_losses = {}'.format(FLAGS.dataset_part, avg_all_losses))
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


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
