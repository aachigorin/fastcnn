import os
import scipy.misc
import skimage.draw
import numpy as np

import tensorflow as tf

from model import MtcnnOnet
from dataset.aflw_reader import AFLWReader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_examples', '1',
                           """.""")
tf.app.flags.DEFINE_string('gpus', '0',
                           'Available gpus')
tf.app.flags.DEFINE_string('save_to', 'faces/',
                           'Path to the directory for output images')


def main(argv=None):  # pylint: disable=unused-argument
  def reader():
    def simple_preprocessor(image):
      with tf.name_scope('random_simple_preprocess'):
        image = image - 0.5
        image = tf.image.resize_images(image, size=[48, 48])
      return image

    return AFLWReader(data_dir=FLAGS.aflw_data_dir,
                        batch_size=1,
                        part=AFLWReader.DatasetPart.test,
                        processor=simple_preprocessor)

  def model():
    return MtcnnOnet()

  with tf.name_scope('tester_reader') as scope:
    reader = reader()
    images = reader.get_batch()
    reader_summaries = tf.summary.merge(
      tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

  model = model()
  predictions = model.inference(images, is_train=False)

  saver = tf.train.Saver()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = FLAGS.gpus

  with tf.Session(config=config) as sess:
    reader.init(sess)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(os.path.abspath(FLAGS.train_dir))
    global_step = _get_global_step(ckpt)
    assert(global_step is not None)
    saver.restore(sess, ckpt.model_checkpoint_path)

    for i in xrange(FLAGS.num_examples):
      img_val, predictions_val = sess.run([images, predictions])
      img_val = np.squeeze(img_val)
      radius = 2
      img_to_save = _draw_points(img_val, predictions_val, color=[1, 1, 1], radius=radius,
                                 method=skimage.draw.circle)
      scipy.misc.imsave(os.path.join(FLAGS.save_to, '{}_img_orig.jpg'.format(i)), img_val)
      scipy.misc.imsave(os.path.join(FLAGS.save_to, '{}_img.jpg'.format(i)), img_to_save)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


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


def _draw_points(img, points, color, radius=2, method=skimage.draw.circle):
  img_to_save = np.copy(img)
  h = img_to_save.shape[0]
  w = img_to_save.shape[1]
  n_points = points.shape[1] / 2
  for i_point in xrange(n_points):
    c, r = points[0, i_point], points[0, n_points + i_point]
    rr, cc = method(np.int32(r * h), np.int32(c * w), radius, shape=(h, w))
    img_to_save[rr, cc, :] = color
  return img_to_save


if __name__ == '__main__':
  tf.app.run()