from __future__ import print_function

import os
import glob

import scipy.io
import scipy.misc
import numpy as np
import skimage.draw

import tensorflow as tf

from dataset.reader import BaseReader


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('celeba_data_dir', '/media/p.omenitsch/code/facedet/MTCNN_train/ONetOut/',
                           """Path to the Celeba data, preprocessed by Phillip""")
tf.app.flags.DEFINE_integer('max_num_parts', '20',
                        """Maximum number of parts from the dataset to take""")


class CelebaReader(BaseReader):
  class DatasetPart(object):
    train = 'train'
    val = 'val'
    test = 'test'


  def __init__(self, data_dir, batch_size, part, processor=None):
    print('Started loading dataset')
    self.batch_size = batch_size
    self.processor = processor
    self.part = part
    self.num_preprocess_threads = 10
    self.min_queue_examples = 10 * batch_size

    if self.part == CelebaReader.DatasetPart.train:
      name_pattern = 'celebagt2data_x[0-9]*.mat'
    elif self.part == CelebaReader.DatasetPart.test:
      name_pattern = 'celebagt2data_[0-9]*.mat'
    else:
      raise Exception("Unsupported dataset part {}".format(part))

    # TODO: can move the loading part into init method and derive just shapes and lenght here
    self.points = []
    self.imgs = []
    for file_idx, fpath in enumerate(glob.glob(os.path.join(data_dir, name_pattern))):
      print('Loading dataset part #{}. File {}'.format(file_idx, fpath))
      if file_idx >= FLAGS.max_num_parts:
        break

      mat = scipy.io.loadmat(fpath)
      points = mat['pointss']
      imgs = mat['imgs']
      is_face = mat['face'] == 1
      is_face = is_face.reshape(mat['face'].shape[0])
      faces_to_take = np.logical_and(is_face,
                                     np.logical_and(np.amax(points, 1) <= 1, np.amin(points, 1) >= 0))

      points = points[faces_to_take, :]
      imgs = np.transpose(imgs[:, :, :, faces_to_take], (3, 0, 1, 2))

      self.points.append(points)
      self.imgs.append(imgs)
    print('Finished loading dataset')


  def get_batch(self):
    print('Number of parts for tf.train.batch_join = {}'.format(len(self.points)))
    n_samples = 0
    for idx, p in enumerate(self.points):
      n_rows = int(p.shape[0])
      n_samples += int(n_rows)
      print('Number of samples in the part #{} = {}'.format(idx, n_rows))
    print('Total number of samples = {}'.format(n_samples))

    parts = []
    self.imgs_placeholder = []
    self.labels_placeholder = []
    self.tf_imgs = []
    self.tf_labels = []
    for i in xrange(len(self.imgs)):
      self.imgs_placeholder.append(tf.placeholder(dtype=tf.float32, shape=self.imgs[i].shape))
      self.labels_placeholder.append(tf.placeholder(dtype=tf.float32, shape=self.points[i].shape))
      self.tf_imgs.append(tf.Variable(self.imgs_placeholder[i], trainable=False, collections=[]))
      self.tf_labels.append(tf.Variable(self.labels_placeholder[i], trainable=False, collections=[]))

      #if self.processor is not None:
      #  tf_imgs = self.processor(tf_imgs)

      tf_imgs, tf_labels = \
        tf.train.slice_input_producer([self.tf_imgs[i], self.tf_labels[i]],
                                      shuffle=True if self.part == CelebaReader.DatasetPart.train else False,
                                      capacity=self.batch_size * 5)
      parts.append((tf_imgs, tf_labels))

    print('Number of parts in celeba {}'.format(len(parts)))
    return tf.train.batch_join(
          parts,
          batch_size=self.batch_size,
          capacity=self.min_queue_examples + 5 * self.batch_size)


  def init(self, sess):
    print('Initializing the data')
    for i in xrange(len(self.imgs_placeholder)):
      sess.run(self.tf_imgs[i].initializer,
               feed_dict={self.imgs_placeholder[i]: self.imgs[i]})
      sess.run(self.tf_labels[i].initializer,
               feed_dict={self.labels_placeholder[i]: self.points[i]})
    print('Finished initializing the data')

if __name__ == '__main__':
  reader = CelebaReader(FLAGS.celeba_data_dir, processor=None, batch_size=10,
               part=CelebaReader.DatasetPart.train)
  batch = reader.get_batch()
  print(batch)