from reader import Cifar10Reader
from model import Cifar10Resnet18

import tensorflow as tf

import os
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_examples', '1000',
                           """.""")
tf.app.flags.DEFINE_string('dataset_part', 'val',
                           """.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', '300',
                           """.""")
tf.app.flags.DEFINE_bool('run_once', False,
                           """.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                           """.""")

class Tester(object):
  def __init__(self, name, data_dir, model_dir, dataset_part, num_examples):
    self.model_dir = model_dir
    self.num_examples = num_examples

    with tf.name_scope(name):
      with tf.name_scope('reader'):
        val_reader = Cifar10Reader(data_dir=data_dir, batch_size=FLAGS.batch_size,
                               part=dataset_part,
                               preprocessing=Cifar10Reader.Preprocessing.simple)
        val_images, val_labels = val_reader.get_batch()

      with tf.name_scope('model') as scope:
        model = Cifar10Resnet18()
        logits = model.inference(val_images, is_train=False)
        self.total_loss, self.top1 = model.loss(logits, val_labels) 

      self.saver = tf.train.Saver() 


  def eval(self):
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess,
                  collection=tf.GraphKeys.QUEUE_RUNNERS, coord=coord)

      ckpt = tf.train.get_checkpoint_state(os.path.abspath(self.model_dir))
      if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(sess, ckpt.model_checkpoint_path)

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')
        return

      num_iters = FLAGS.num_examples // FLAGS.batch_size
      count_loss = 0.
      count_top1 = 0.

      for i in xrange(num_iters):
        loss_val, top1_val = sess.run([self.total_loss, self.top1])
        count_loss += loss_val
        count_top1 += top1_val

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
      return count_loss / num_iters, count_top1 / num_iters


def main(unused_argv=None):
  tester = Tester('', FLAGS.data_dir, FLAGS.train_dir,
                  FLAGS.dataset_part, FLAGS.num_examples)
  while True:
    loss, top1 = tester.eval()
    print('{} results. total_loss = {}, top1 accuracy = {}'.format(
          FLAGS.dataset_part, loss, top1))
    time.sleep(FLAGS.eval_interval_secs)
    if FLAGS.run_once:
      break


if __name__ == '__main__':
  tf.app.run()
