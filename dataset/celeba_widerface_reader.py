from __future__ import print_function

import tensorflow as tf

import celeba_reader
import widerface_reader

from dataset.reader import BaseReader

FLAGS = tf.app.flags.FLAGS


class CelebaWiderfaceReader(BaseReader):
  class DatasetPart(object):
    train = 'train'
    val = 'val'
    test = 'test'


  def __init__(self, batch_size, part):
    print('Started loading dataset')
    self.batch_size = batch_size
    self.part = part
    self.num_preprocess_threads = 10
    self.min_queue_examples = 10 * batch_size

    def preprocessor_celeba(image):
      with tf.name_scope('preprocessor_celeba'):
        image = image - 0.5
        #image = tf.image.central_crop(image, 0.88) # 54 -> 48
        #image = tf.reshape(image, [48, 48, 3])
        image = tf.image.resize_images(image, [48, 48])
      return image

    def preprocessor_widerface(image):
      with tf.name_scope('preprocessor_widerface'):
        image = image - 0.5
      return image

    self.celeba = celeba_reader.CelebaReader(FLAGS.celeba_data_dir,
                                             batch_size, part, processor=preprocessor_celeba)
    self.widerface = widerface_reader.WiderfaceReader(FLAGS.celeba_data_dir,
                                                      batch_size, part, processor=preprocessor_widerface)
    print('Finished loading dataset')


  def get_batch(self):
    self.celeba.get_batch()
    self.widerface.get_batch()
    parts_faces, parts_partfaces, parts_not_faces = self.widerface.get_parts()
    parts_landmarks = self.celeba.get_parts()

    assert(self.batch_size % 4 == 0)
    bs = self.batch_size / 4
    faces_batch = tf.train.batch_join(parts_faces, batch_size=bs,
                               capacity=self.min_queue_examples + 5 * bs)
    partfaces_batch = tf.train.batch_join(parts_partfaces, batch_size=bs,
                                 capacity=self.min_queue_examples + 5 * bs)
    not_faces_batch = tf.train.batch_join(parts_not_faces, batch_size=bs,
                                 capacity=self.min_queue_examples + 5 * bs)
    landmarks_batch = tf.train.batch_join(parts_landmarks, batch_size=bs,
                                 capacity=self.min_queue_examples + 5 * bs)

    return [tf.concat([faces_batch[i], partfaces_batch[i],
                       not_faces_batch[i], landmarks_batch[i]], axis=0) for i in range(0,2)]
    #return tf.train.batch_join(parts, batch_size=self.batch_size,
    #                           capacity=self.min_queue_examples + 5 * self.batch_size)


  def init(self, sess):
    print('Initializing the data')
    self.celeba.init(sess)
    self.widerface.init(sess)
    print('Finished initializing the data')


if __name__ == '__main__':
  reader = CelebaWiderfaceReader(batch_size=10, part=CelebaWiderfaceReader.DatasetPart.train)
  batch = reader.get_batch()
  print(batch)