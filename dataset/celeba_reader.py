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
tf.app.flags.DEFINE_string('celeba_data_dir', '',
                           """Path to the Celeba data, preprocessed by Phillip""")


class CelebaReader(BaseReader):
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

    if self.part == CelebaReader.DatasetPart.train:
      name_pattern = 'celebagt2data_x*.mat'
    elif self.part == CelebaReader.DatasetPart.test:
      name_pattern = 'celebagt2data_*.mat'
    else:
      raise Exception("Unsupported dataset part {}".format(part))

    self.points = None
    self.imgs = None
    for file_idx, fpath in enumerate(glob.glob(os.path.join(data_dir, name_pattern))):
      if file_idx >= 2:
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

      if self.points is None:
        self.points = points
        self.imgs = imgs
      else:
        self.points = np.concatenate((self.points, points))
        self.imgs = np.concatenate((self.imgs, imgs))

      # if file_idx == 1:
      #   for img_idx in xrange(self.imgs.shape[3]):
      #     if img_idx > 10:
      #       break
      #
      #     cur_points = self.points[img_idx, :]
      #     img_to_save = np.copy(self.imgs[:,:,:,img_idx])
      #     n_points = len(cur_points) / 2
      #     for i_point in xrange(n_points):
      #       c, r = cur_points[i_point], cur_points[n_points + i_point]
      #       rr, cc = skimage.draw.circle(r * img_to_save.shape[0], c * img_to_save.shape[1], 2)
      #       img_to_save[rr, cc, :] = 1
      #     scipy.misc.imsave('faces/face_with_points_{}.jpg'.format(img_idx), img_to_save)


  def get_batch(self):
    tf_imgs = tf.constant(self.imgs)
    tf_labels = tf.constant(self.points)

    if self.processor is not None:
      tf_imgs = self.processor(tf_imgs)

    if self.part == CelebaReader.DatasetPart.train:
      tf_imgs, tf_labels = \
        tf.train.slice_input_producer([tf_imgs, tf_labels],
                                      shuffle = True if self.part == CelebaReader.DatasetPart.train else False)

    return tf.train.batch(
            [tf_imgs, tf_labels],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=self.min_queue_examples + 3 * self.batch_size)


if __name__ == '__main__':
  reader = CelebaReader(FLAGS.celeba_data_dir, processor=None, batch_size=10,
               part=CelebaReader.DatasetPart.train)
  batch = reader.get_batch()
  print(batch)