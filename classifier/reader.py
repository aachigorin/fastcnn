from __future__ import print_function

from abc import ABCMeta, abstractmethod
import os
import scipy.misc

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

class GenericReader(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_batch():
    pass


class Cifar10Reader(GenericReader):
  HEIGHT = 32
  WIDTH = 32

  class DatasetPart(object):
    train = 'train'
    val = 'val'
    test = 'test'


  class Preprocessing(object):
    simple = 'simple'
    random_simple = 'random_simple'


  def __init__(self, data_dir, batch_size, part=DatasetPart.train,
      preprocessing=Preprocessing.simple):
    self.data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    self.batch_size = batch_size
    self.part = part
    self.preprocessing = preprocessing
    self.train_val_ratio = 0.9
    Cifar10Reader._maybe_download_and_extract(data_dir) 


  def get_batch(self):
    if self.part == Cifar10Reader.DatasetPart.train:
      filenames = [os.path.join(self.data_dir, 'data_batch_%d.bin' % i)
                   for i in xrange(1, 6)]
      train_val_split_idx = int(len(filenames) * self.train_val_ratio)
      filenames = filenames[:train_val_split_idx]
    elif self.part == Cifar10Reader.DatasetPart.val:
      filenames = [os.path.join(self.data_dir, 'data_batch_%d.bin' % i)
                   for i in xrange(1, 6)]
      train_val_split_idx = int(len(filenames) * self.train_val_ratio)
      filenames = filenames[train_val_split_idx:]
    elif self.part == Cifar10Reader.DatasetPart.test:
      filenames = [os.path.join(self.data_dir, 'test_batch.bin')]
    else:
        raise Exception('Unknown dataset part {}'.format(self.part))

    for f in filenames:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    # queue with filenames
    filename_queue = tf.train.string_input_producer(filenames,
                      capacity=self.batch_size * 3)

    # read examples from files in the filename queue.
    read_input = Cifar10Reader._read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    if self.preprocessing == Cifar10Reader.Preprocessing.simple:
      prepcocessed_image = self.simple_preprocess(reshaped_image)
    elif self.preprocessing == Cifar10Reader.Preprocessing.random_simple:
      prepcocessed_image = self.random_simple_preprocess(reshaped_image)
    else:
      raise Exception('Unknown preprocessing {}'.format(self.preprocessing))

    do_shuffle = self.part == Cifar10Reader.DatasetPart.train
    return self._generate_image_and_label_batch(
            prepcocessed_image, read_input.label,
            self.batch_size * 10, self.batch_size, shuffle=do_shuffle)


  def simple_preprocess(self, image):
    with tf.name_scope('simple_preprocess'):
      image = tf.image.resize_image_with_crop_or_pad(image,
                        Cifar10Reader.WIDTH, Cifar10Reader.HEIGHT)
      image = tf.image.per_image_whitening(image)
    return image


  def random_simple_preprocess(self, image):
    with tf.name_scope('random_simple_preprocess'):
      image = tf.image.resize_image_with_crop_or_pad(image,
                        Cifar10Reader.WIDTH + 2 * 4, Cifar10Reader.HEIGHT + 2 * 4)
      image = tf.random_crop(image, [Cifar10Reader.WIDTH, Cifar10Reader.HEIGHT, 3])
      image = tf.image.random_flip_left_right(image)
      image = tf.image.per_image_whitening(image)
    return image


  def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                      batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 1
    if shuffle:
      images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
      images, label_batch = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)

    # Display the images in the visualizer.
    tf.image_summary(self.part + '_images', images)

    return images, tf.reshape(label_batch, [batch_size])


  @staticmethod
  def _read_cifar10(filename_queue):
    class CIFAR10Record(object):
      pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


  @staticmethod
  def _maybe_download_and_extract(data_dir):
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)
