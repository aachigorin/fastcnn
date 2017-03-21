from __future__ import print_function

import os
import glob

import scipy.io
import scipy.misc
import numpy as np
import skimage.draw

import tensorflow as tf
from tensorflow.contrib import lookup

from dataset.reader import BaseReader


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('aflw_data_dir', '/media/p.omenitsch/code/facedet/MTCNN_face_detection_alignment/code/codes/MTCNNv1/',
                           """Path to the Celeba data, preprocessed by Phillip""")
tf.app.flags.DEFINE_integer('max_num_parts', '20',
                        """Maximum number of parts from the dataset to take""")
tf.app.flags.DEFINE_integer('resize_to', '54',
                        """Maximum number of parts from the dataset to take""")

class AFLWReader(BaseReader):
  class DatasetPart(object):
    train = 'train'
    val = 'val'
    test = 'test'


  def __init__(self, data_dir, batch_size, part, processor=None):
    self.batch_size = batch_size
    self.processor = processor
    self.part = part
    self.num_preprocess_threads = 10
    self.min_queue_examples = 10 * batch_size

    if self.part == AFLWReader.DatasetPart.test:
      name_pattern = 'boundingboxesAFLW*.mat'
    else:
      raise Exception("Unsupported dataset part {}".format(part))

    self.paths = []
    self.bboxes = []
    self.path2bbox_idx = []
    for file_idx, fpath in enumerate(glob.glob(os.path.join(data_dir, name_pattern))):
      if file_idx >= FLAGS.max_num_parts:
        break

      mat = scipy.io.loadmat(fpath)
      img_names = np.transpose(mat['imnames'], (1, 0))
      bboxes = mat['total_boxes']
      good_boxes_idx = np.logical_and(np.logical_and(bboxes[:, 2] - bboxes[:, 0] > 40,
                                                     bboxes[:, 3] - bboxes[:, 1] > 40),
                                      bboxes[:, 4] > 0.8)
      good_boxes_idx = np.logical_and(good_boxes_idx, bboxes[:, 0] >= 0)
      good_boxes_idx = np.logical_and(good_boxes_idx, bboxes[:, 1] >= 0)

      img_names = img_names[good_boxes_idx]
      bboxes = bboxes[good_boxes_idx]
      print(len(good_boxes_idx), len(bboxes))

      paths = []
      for idx, path in enumerate(img_names):
        paths.append(path[0][0])

      self.bboxes.append(bboxes)
      self.paths.append(paths)
      self.path2bbox_idx.append(lookup.HashTable(
        lookup.KeyValueTensorInitializer(paths, range(0, len(bboxes))),
        default_value=-1))


  def get_batch(self):
    parts = []
    self.paths_placeholder = []
    self.bboxes_placeholder = []
    self.tf_paths = []
    self.tf_bboxes = []
    for i in xrange(len(self.paths)):
      self.paths_placeholder.append(tf.placeholder(dtype=tf.string, shape=len(self.paths[i])))
      self.tf_paths.append(tf.Variable(self.paths_placeholder[i], trainable=False, collections=[]))
      self.tf_bboxes = tf.constant(self.bboxes[i], dtype=tf.int32)

      #if self.processor is not None:
      #  tf_path = self.processor(tf_path)

      tf_img_path = tf.train.string_input_producer(self.tf_paths[i],
                                                   shuffle=True if self.part == AFLWReader.DatasetPart.train else False)
      tf_img_path, tf_img = tf.WholeFileReader().read(tf_img_path)
      bbox_idx = self.path2bbox_idx[i].lookup(tf_img_path)
      tf_bboxes = self.tf_bboxes[bbox_idx, :]

      tf_img = tf.image.decode_jpeg(tf_img, channels=3)

      h, w = tf.shape(tf_img)[0], tf.shape(tf_img)[1]
      c1, r1, c2, r2 = tf_bboxes[0], tf_bboxes[1], tf_bboxes[2], tf_bboxes[3]
      tf_img = tf.image.crop_to_bounding_box(tf_img, r1, c1,
                                             tf.minimum(h - r1, r2 - r1), tf.minimum(w - c1, c2 - c1))
      tf_img = tf.image.resize_images(tf_img, size=(FLAGS.resize_to, FLAGS.resize_to))
      tf_img = (tf.cast(tf_img, dtype=tf.float32) - 127.5) * 0.0078125
      parts.append([tf_img])

    return tf.train.batch_join(
          parts,
          batch_size=self.batch_size,
          capacity=self.min_queue_examples + 5 * self.batch_size)


  def init(self, sess):
    for i in xrange(len(self.paths_placeholder)):
      sess.run(self.tf_paths[i].initializer,
               feed_dict={self.paths_placeholder[i]: self.paths[i]})

    for table in self.path2bbox_idx:
        sess.run(table.init)


if __name__ == '__main__':
  reader = AFLWReader(FLAGS.aflw_data_dir, processor=None, batch_size=100,
               part=AFLWReader.DatasetPart.test)
  batch = reader.get_batch()

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.3
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = '1'
  sess = tf.Session(config=config)

  reader.init(sess)
  sess.run(tf.global_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  batch_val = sess.run(batch)
  print(batch_val)
  scipy.misc.imsave('test.jpg', batch_val[0])

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)