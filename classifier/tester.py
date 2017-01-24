import os
import time

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
    reader = reader()
    images, labels = reader.get_batch()
    reader_summaries = tf.summary.merge(
      tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

  model = model()
  logits = model.inference(images, is_train=False)
  total_loss, top1 = model.loss(logits, labels)

  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                            tf.get_default_graph())

  while True:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.visible_device_list=FLAGS.gpus

    with tf.Session(config=config) as sess:
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
      count_loss = 0.
      count_top1 = 0.
      for i in xrange(num_iters):
        loss_val, top1_val, reader_summaries_str = sess.run(
          [total_loss, top1, reader_summaries])
        count_loss += loss_val
        count_top1 += top1_val
      avg_total_loss = count_loss / num_iters
      avg_top1_acc = count_top1 / num_iters

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

      # adding summaries
      summary_writer.add_summary(reader_summaries_str, global_step)

      summary = tf.Summary()
      summary.value.add(tag='tester/avg_total_loss', simple_value=avg_total_loss)
      summary.value.add(tag='tester/avg_top1_accuracy', simple_value=avg_top1_acc)
      summary_writer.add_summary(summary, global_step)

    print('{} results. total_loss = {}, avg_top1_accuracy = {}'.format(
          FLAGS.dataset_part, avg_total_loss, avg_top1_acc))
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
