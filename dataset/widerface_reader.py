from __future__ import print_function

import os
import glob

import scipy.io
import scipy.misc
import numpy as np
import skimage.draw

import classifier.tf_utils as tf_utils
import tensorflow as tf

from dataset.reader import BaseReader


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('widerface_data_dir', '/media/p.omenitsch/code/facedet/MTCNN_train/ONetOut/',
                           """Path to the Widerface data, preprocessed by Phillip""")
tf.app.flags.DEFINE_integer('widerface_max_num_parts', '20',
                        """Maximum number of parts from the dataset to take""")


class WiderfaceReader(BaseReader):
  class DatasetPart(object):
    train = 'train'
    val = 'val'
    test = 'test'


  def __init__(self, data_dir, batch_size, part, processor=None):
    print('Started loading dataset Widerface from {}'.format(data_dir))
    self.batch_size = batch_size
    self.processor = processor
    self.part = part
    self.num_preprocess_threads = 10
    self.min_queue_examples = 10 * batch_size

    if self.part == WiderfaceReader.DatasetPart.train:
      name_pattern = 'widergt3data_[0-9]*.mat'
      self.files_paths = sorted(glob.glob(os.path.join(data_dir, name_pattern)))
    elif self.part == WiderfaceReader.DatasetPart.test:
      name_pattern = 'widergt3testdata_[0-9]*.mat'
      self.files_paths = sorted(glob.glob(os.path.join(data_dir, name_pattern)))
    else:
      raise Exception("Unsupported dataset part {}".format(part))

    # TODO: can move the loading part into init method and derive just shapes and length here
    self._imgs_faces_shape = []
    self._imgs_not_faces_shape = []
    self._imgs_partfaces_shape = []
    self._labels_faces = []
    self._labels_not_faces = []
    self._labels_partfaces = []
    for file_idx, fpath in enumerate(self.files_paths):
      if file_idx >= FLAGS.widerface_max_num_parts:
        break
      print('Loading dataset part #{}. File {}'.format(file_idx, fpath))

      imgs_faces, imgs_not_faces, imgs_partfaces, bboxes_faces, bboxes_partfaces = WiderfaceReader._load_mat_file(fpath)
      print('Finish loading matrices')

      print('faces', imgs_faces.shape)
      print('not_faces', imgs_not_faces.shape)
      print('partfaces', imgs_partfaces.shape)

      self._imgs_faces_shape.append(imgs_faces.shape)
      self._imgs_not_faces_shape.append(imgs_not_faces.shape)
      self._imgs_partfaces_shape.append(imgs_partfaces.shape)

      # labels:
      # 1 - cost type (0 - not face, 1 - face, 2 - partface, 3 - landmarks)
      # 1 - face / not face
      # 4 - bbox regression
      # 10 - landmarks
      labels_faces = np.zeros(shape=(imgs_faces.shape[0], 16))
      labels_faces[:, 0] = 1 # cost type
      labels_faces[:, 1] = 1 # face or not
      labels_faces[:, 2:6] = bboxes_faces[:, :]
      self._labels_faces.append(labels_faces)

      labels_partfaces = np.zeros(shape=(imgs_partfaces.shape[0], 16))
      labels_partfaces[:, 0] = 2  # cost type
      labels_partfaces[:, 1] = 0  # face or not
      labels_partfaces[:, 2:6] = bboxes_partfaces[:, :]
      self._labels_partfaces.append(labels_partfaces)

      labels_not_faces = np.zeros(shape=(imgs_not_faces.shape[0], 16))
      labels_not_faces[:, 0] = 0  # cost type
      labels_not_faces[:, 1] = 0  # face or not
      self._labels_not_faces.append(labels_not_faces)

      # check the results
      def _draw_bbox(img, r1, c1, r2, c2):
        r = [r1, r1, r2, r2]
        c = [c1, c2, c2, c1]
        rr, cc = skimage.draw.polygon_perimeter(r, c, clip=True, shape=(img.shape[0], img.shape[1]))
        img[rr, cc] = [1, 0, 0]
        return img

      def _visualize(prefix, imgs, labels=None, n_imgs=50):
        for i in xrange(n_imgs):
          cur_img = imgs[i, :].copy()
          r1, c1 = cur_img.shape[0], cur_img.shape[1]

          if labels is not None:
            r1 = r1 + labels[i][3] * cur_img.shape[0]
            c1 = c1 + labels[i][2] * cur_img.shape[1]
            r2 = r1 + (1 + labels[i][5]) * cur_img.shape[0]
            c2 = c1 + (1 + labels[i][4]) * cur_img.shape[1]
          else:
            r2 = r1 + cur_img.shape[1]
            c2 = c1 + cur_img.shape[0]

          pad_size = (cur_img.shape[0], cur_img.shape[1])
          cur_img = np.pad(cur_img, (pad_size, pad_size, (0, 0)), mode='constant')

          scipy.misc.imsave(os.path.join('test/{}_{}.jpg'.format(prefix, i)),
                            _draw_bbox(cur_img, r1, c1, r2, c2))

      if file_idx == -1:
        _visualize('img_not_face', imgs_not_faces, n_imgs=50)
        _visualize('img_face', imgs_faces, labels_faces, n_imgs=50)
        _visualize('img_partface', imgs_partfaces, labels_partfaces, n_imgs=50)

    print('Finished loading dataset')


  def get_batch(self):
    print('Number of parts for tf.train.batch_join = {}'.format(len(self._imgs_faces_shape)))
    n_samples = 0
    for idx in xrange(len(self._imgs_faces_shape)):
      n_rows = int(self._imgs_faces_shape[idx][0] + self._imgs_not_faces_shape[idx][0] +
                   self._imgs_partfaces_shape[idx][0])
      n_samples += int(n_rows)
      print('Number of samples in the part #{} = {}'.format(idx, n_rows))
    print('Total number of samples = {}'.format(n_samples))

    self.parts_faces = []
    self.parts_partfaces = []
    self.parts_not_faces = []

    self.imgs_faces_placeholder = []
    self.imgs_partfaces_placeholder = []
    self.imgs_not_faces_placeholder = []

    self.labels_faces_placeholder = []
    self.labels_partfaces_placeholder = []
    self.labels_not_faces_placeholder = []

    self.tf_imgs_faces = []
    self.tf_imgs_partfaces = []
    self.tf_imgs_not_faces = []

    self.tf_labels_faces = []
    self.tf_labels_partfaces = []
    self.tf_labels_not_faces = []

    def process_data_type(imgs_shape, labels):
      imgs_placeholder = tf.placeholder(dtype=tf.float32, shape=imgs_shape)
      labels_placeholder = tf.placeholder(dtype=tf.float32, shape=labels.shape)
      tf_imgs_var = tf.Variable(imgs_placeholder, trainable=False, collections=[])
      tf_labels_var = tf.Variable(labels_placeholder, trainable=False, collections=[])
      tf_imgs, tf_labels = \
      tf.train.slice_input_producer([tf_imgs_var, tf_labels_var],
                                     shuffle=True if self.part == WiderfaceReader.DatasetPart.train else False,
                                     capacity=self.batch_size * 5)

      if self.processor is not None:
        tf_imgs, tf_labels = self.processor(tf_imgs, tf_labels)

      return imgs_placeholder, labels_placeholder, tf_imgs_var, tf_labels_var, tf_imgs, tf_labels

    for i in xrange(len(self._imgs_faces_shape)):
      res = process_data_type(self._imgs_faces_shape[i], self._labels_faces[i])
      self.imgs_faces_placeholder.append(res[0])
      self.labels_faces_placeholder.append(res[1])
      self.tf_imgs_faces.append(res[2])
      self.tf_labels_faces.append(res[3])
      self.parts_faces.append((res[4], res[5]))

      res = process_data_type(self._imgs_partfaces_shape[i], self._labels_partfaces[i])
      self.imgs_partfaces_placeholder.append(res[0])
      self.labels_partfaces_placeholder.append(res[1])
      self.tf_imgs_partfaces.append(res[2])
      self.tf_labels_partfaces.append(res[3])
      self.parts_partfaces.append((res[4], res[5]))

      res = process_data_type(self._imgs_not_faces_shape[i], self._labels_not_faces[i])
      self.imgs_not_faces_placeholder.append(res[0])
      self.labels_not_faces_placeholder.append(res[1])
      self.tf_imgs_not_faces.append(res[2])
      self.tf_labels_not_faces.append(res[3])
      self.parts_not_faces.append((res[4], res[5]))

    print('Number of parts {}'.format(len(self.parts_faces)))
    images_batch, labels_batch = tf.train.batch_join(
          self.parts_faces + self.parts_partfaces + self.parts_not_faces,
          batch_size=self.batch_size,
          capacity=self.min_queue_examples + 5 * self.batch_size)

    if FLAGS.widerface_no_augmentation == False:
      WiderfaceReader._draw_img_and_bbox(images_batch, labels_batch, scope='after_augmentation')

    return images_batch, labels_batch

  def get_parts(self):
    return self.parts_faces, self.parts_partfaces, self.parts_not_faces


  def init(self, sess):
    print('Initializing the data')
    assert(len(self.imgs_faces_placeholder) <= len(self.files_paths))
    for i in xrange(len(self.imgs_faces_placeholder)):
      print('Initializing from dataset part #{}. File {}'.format(i, self.files_paths[i]))
      imgs_faces, imgs_not_faces, imgs_partfaces, _, _ = WiderfaceReader._load_mat_file(self.files_paths[i])

      sess.run(self.tf_imgs_faces[i].initializer,
               feed_dict={self.imgs_faces_placeholder[i]: imgs_faces})

      sess.run(self.tf_labels_faces[i].initializer,
               feed_dict={self.labels_faces_placeholder[i]: self._labels_faces[i]})

      sess.run(self.tf_imgs_partfaces[i].initializer,
               feed_dict={self.imgs_partfaces_placeholder[i]: imgs_partfaces})

      sess.run(self.tf_labels_partfaces[i].initializer,
               feed_dict={self.labels_partfaces_placeholder[i]: self._labels_partfaces[i]})

      sess.run(self.tf_imgs_not_faces[i].initializer,
               feed_dict={self.imgs_not_faces_placeholder[i]: imgs_not_faces})

      sess.run(self.tf_labels_not_faces[i].initializer,
               feed_dict={self.labels_not_faces_placeholder[i]: self._labels_not_faces[i]})
    print('Finished initializing the data')


  @staticmethod
  def _load_mat_file(fpath):
    mat = scipy.io.loadmat(fpath)

    imgs = mat['imgs']
    # dx, dy, dw, dh
    bboxes = mat['bboxes']

    not_face = mat['face'] == 0
    is_face = mat['face'] == 1
    is_partface = mat['face'] == 2

    to_shape = mat['face'].shape[0]
    is_face = is_face.reshape(to_shape)
    not_face = not_face.reshape(to_shape)
    is_partface = is_partface.reshape(to_shape)

    imgs_faces = np.transpose(imgs[:, :, :, is_face], (3, 0, 1, 2))
    imgs_not_faces = np.transpose(imgs[:, :, :, not_face], (3, 0, 1, 2))
    imgs_partfaces = np.transpose(imgs[:, :, :, is_partface], (3, 0, 1, 2))
    bboxes_faces = bboxes[is_face, :]
    bboxes_partfaces = bboxes[is_partface, :]

    return imgs_faces, imgs_not_faces, imgs_partfaces, bboxes_faces, bboxes_partfaces


  @staticmethod
  def _draw_img_and_bbox(imgs, labels, scope, n_examples=10):
    with tf.name_scope(scope):
      img_size = tf.cast(imgs.get_shape()[1], dtype=tf.float32)
      pad = tf_utils.const(24)

      images = imgs[:n_examples, :, :, :]
      images = tf.pad(images, paddings=[[0,0], tf.concat([pad,pad], axis=0), tf.concat([pad,pad], axis=0), [0,0]])

      bboxes = labels[:n_examples, 2:6]
      x1 = tf.cast(pad, dtype=tf.float32) + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0] # + img_size * bboxes[:, 0]
      y1 = tf.cast(pad, dtype=tf.float32) + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0] # + img_size + img_size * bboxes[:, 2]
      x2 = tf.cast(pad, dtype=tf.float32) + img_size + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0] # * bboxes[:, 1]
      y2 = tf.cast(pad, dtype=tf.float32) + img_size + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0] # + img_size * bboxes[:, 3]
      x1 = tf_utils.const(0.4) + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0]
      y1 = tf_utils.const(0.6) + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0]
      x2 = tf_utils.const(0.4) + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0]
      y2 = tf_utils.const(0.6) + tf_utils.const(0, dtype=tf.float32) * bboxes[:, 0]

      print('tf.stack([x1, y1, x2, y2], axis=0)', tf.stack([x1, y1, x2, y2], axis=0))
      print('tf.stack([x1, y1, x2, y2], axis=1)', tf.stack([x1, y1, x2, y2], axis=1))
      bboxes_to_draw = tf.reshape(tf.stack([x1, y1, x2, y2], axis=1), shape=[n_examples,1,4])
      images = tf.image.draw_bounding_boxes(images, bboxes_to_draw)
      tf.summary.image('images', images, collections=[BaseReader.READER_DEBUG_COLLECTION],
                       max_outputs=n_examples)


if __name__ == '__main__':
  FLAGS.max_num_parts = 1
  reader = WiderfaceReader(FLAGS.widerface_data_dir, processor=None, batch_size=20,
               part=WiderfaceReader.DatasetPart.train)
  batch = reader.get_batch()
  print(batch)

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
  for i in xrange(len(batch_val[0])):
    scipy.misc.imsave('test/test_{}.jpg'.format(i), batch_val[0][i])
    print(i, batch_val[1][i][:2])

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)