from __future__ import print_function

import tensorflow as tf

import classifier.tf_utils as tf_utils
import celeba_reader
import widerface_reader

from dataset.reader import BaseReader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('faces_prop', '1',
                        """Proportion of faces in the batch""")
tf.app.flags.DEFINE_integer('not_faces_prop', '1',
                        """.""")
tf.app.flags.DEFINE_integer('partfaces_prop', '1',
                        """.""")
tf.app.flags.DEFINE_integer('landmarks_prop', '1',
                        """.""")
tf.app.flags.DEFINE_boolean('widerface_no_augmentation', 'True',
                        """.""")
tf.app.flags.DEFINE_boolean('celeba_no_augmentation', 'True',
                        """.""")


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

    def preprocessor_celeba(image, labels):
      ins = 54  # input_size
      ts = 48  # target_size
      with tf.name_scope('preprocessor_celeba'):
        if FLAGS.celeba_no_augmentation:
          image = tf.image.resize_images(image, [48, 48])
          image = image - 0.5
        else:
          # random crop starting pixel coords
          r_p_crop = tf.random_uniform(shape=[1, 1], minval=0, maxval=ins - ts)
          c_p_crop = tf.random_uniform(shape=[1, 1], minval=0, maxval=ins - ts)
          image = tf.slice(image, [r_p_crop, c_p_crop, tf_utils.const(0)], [ts, ts, tf_utils.const(-1)])
          image = tf.reshape(image, shape=[ts, ts, 3])
          image = image - 0.5

          n_points = 5
          r_range = xrange(6, 6+n_points)
          c_range = xrange(6+n_points, 6+2*n_points)
          r_p = ins * labels[r_range]
          c_p = ins * labels[c_range]

          r_p_new = (r_p - r_p_crop) / ts
          c_p_new = (c_p - c_p_crop) / ts
          labels[r_range] = r_p_new
          labels[c_range] = c_p_new
      return image, labels


    def preprocessor_widerface(image, labels):
      ins = 54 # input_size
      ts = 48 # target_size
      with tf.name_scope('preprocessor_widerface'):
        if FLAGS.widerface_no_augmentation:
          image = tf.image.resize_images(image, [ts, ts])
          image = image - 0.5
        else:
          print('image', image)
          print('labels', labels)
          # random crop starting pixel coords
          r_p_crop = tf.random_uniform(shape=[1], minval=0, maxval=ins-ts, dtype=tf.int32)
          c_p_crop = tf.random_uniform(shape=[1], minval=0, maxval=ins-ts, dtype=tf.int32)
          image = tf.slice(image, tf_utils.stack_and_squeeze([r_p_crop, c_p_crop, tf_utils.const(0)]),
                           tf_utils.stack_and_squeeze([tf_utils.const(ts), tf_utils.const(ts), tf_utils.const(-1)]))
          image = tf.reshape(image, shape=[ts, ts, 3])
          image = image - 0.5

          # current shifts in pixels
          bbox = tf.multiply(tf.cast(ins, dtype=tf.float32), labels[2:6])
          dr_p, dc_p, dh_p, dw_p = bbox[0], bbox[1], bbox[2], bbox[3]

          # new shift relative to the crop
          ts_float = tf.cast(ts, dtype=tf.float32)
          dr_new = tf.reshape((dr_p - tf.cast(r_p_crop, dtype=tf.float32)) / ts_float, shape=[1])
          dc_new = tf.reshape((dc_p - tf.cast(c_p_crop, dtype=tf.float32)) / ts_float, shape=[1])
          dh_new = tf.reshape((ts_float - dh_p) / ts_float, shape=[1])
          dw_new = tf.reshape((ts_float - dw_p) / ts_float, shape=[1])
          print('labels[0:2]', labels[0:2])
          print('labels[6:]', labels[6:])
          labels = tf.concat([labels[0:2], dr_new, dc_new, dh_new, dw_new, labels[6:]], axis=0)

          print('labels', labels)
          print('image', image)
          #labels[2] = dr_new
          #labels[3] = dc_new
          #labels[4] = dh_new
          #labels[5] = dw_new
      return image, labels


    self.celeba = celeba_reader.CelebaReader(FLAGS.celeba_data_dir,
                                             batch_size, part, processor=preprocessor_celeba)
    self.widerface = widerface_reader.WiderfaceReader(FLAGS.widerface_data_dir,
                                                      batch_size, part, processor=preprocessor_widerface)
    print('Finished loading dataset')


  def get_batch(self):
    self.celeba.get_batch()
    self.widerface.get_batch()
    parts_faces, parts_partfaces, parts_not_faces = self.widerface.get_parts()
    parts_landmarks = self.celeba.get_parts()

    num_parts = FLAGS.faces_prop + FLAGS.not_faces_prop + FLAGS.partfaces_prop + FLAGS.landmarks_prop
    assert(self.batch_size % num_parts == 0)

    bs = self.batch_size / num_parts
    faces_bs = bs * FLAGS.faces_prop
    faces_batch = tf.train.batch_join(parts_faces, batch_size=faces_bs,
                               capacity=self.min_queue_examples + 5 * faces_bs)

    partfaces_bs = bs * FLAGS.partfaces_prop
    partfaces_batch = tf.train.batch_join(parts_partfaces, batch_size=partfaces_bs,
                                 capacity=self.min_queue_examples + 5 * partfaces_bs)

    not_faces_bs = bs * FLAGS.not_faces_prop
    not_faces_batch = tf.train.batch_join(parts_not_faces, batch_size=not_faces_bs,
                                 capacity=self.min_queue_examples + 5 * not_faces_bs)

    landmarks_bs = bs * FLAGS.landmarks_prop
    landmarks_batch = tf.train.batch_join(parts_landmarks, batch_size=landmarks_bs,
                                 capacity=self.min_queue_examples + 5 * landmarks_bs)

    return [tf.concat([faces_batch[i], partfaces_batch[i],
                       not_faces_batch[i], landmarks_batch[i]], axis=0) for i in range(0,2)]


  def init(self, sess):
    print('Initializing the data')
    self.celeba.init(sess)
    self.widerface.init(sess)
    print('Finished initializing the data')


if __name__ == '__main__':
  reader = CelebaWiderfaceReader(batch_size=10, part=CelebaWiderfaceReader.DatasetPart.train)
  batch = reader.get_batch()
  print(batch)