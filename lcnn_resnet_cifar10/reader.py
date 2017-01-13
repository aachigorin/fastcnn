from __future__ import print_function

import os
import sys
import urllib

import scipy.misc
import tensorflow as tf

from fastcnn.classifier.reader import BaseReader


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")


class Cifar10Reader(BaseReader):
  HEIGHT = 32
  WIDTH = 32

  MEAN = [125.3, 123.0, 113.9]
  STD = [63.0,  62.1,  66.7]

  class DatasetPart(object):
    train = 'train'
    val = 'val'
    test = 'test'


  class Preprocessing(object):
    simple = 'simple'
    random_simple = 'random_simple'


  def __init__(self, data_dir, batch_size, processor, part):
    #  preprocessing=Preprocessing.simple):
    self.data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    self.batch_size = batch_size
    self.part = part
    #self.preprocessing = preprocessing
    Cifar10Reader._maybe_download_and_extract(data_dir) 

    if self.part == Cifar10Reader.DatasetPart.train:
      filenames = [os.path.join(self.data_dir, 'data_batch_%d.bin' % i)
                   for i in xrange(1, 6)]
      self.num_preprocess_threads = 16
    elif self.part == Cifar10Reader.DatasetPart.val:
        raise "no val part in cifar10"
    elif self.part == Cifar10Reader.DatasetPart.test:
      filenames = [os.path.join(self.data_dir, 'test_batch.bin')]
      self.num_preprocess_threads = 1 # to prevent random behaviour
    else:
        raise Exception('Unknown dataset part {}'.format(self.part))

    for f in filenames:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    # queue with filenames
    filename_queue = tf.train.string_input_producer(filenames,
                      capacity=self.batch_size * 3)

    # read examples from files in the filename queue.
    read_input = self._read_cifar10(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    preprocessed_image = processor(image)
    #if self.preprocessing == Cifar10Reader.Preprocessing.simple:
    #  preprocessed_image = self.simple_preprocess(image)
    #elif self.preprocessing == Cifar10Reader.Preprocessing.random_simple:
    #  preprocessed_image = self.random_simple_preprocess(image)
    #else:
    #  raise Exception('Unknown preprocessing {}'.format(self.preprocessing))

    #self.raw_img = preprocessed_image
    do_shuffle = self.part == Cifar10Reader.DatasetPart.train
    self.result_batch = self._generate_image_and_label_batch(
            preprocessed_image, read_input.label,
            self.batch_size * 10, self.batch_size, shuffle=do_shuffle)

    self.reset_reader_op = self.reader.reset()


  def get_reset_reader_op(self):
    return self.reset_reader_op


  def get_batch(self):
    return self.result_batch


  def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                      batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    if shuffle:
      images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=self.num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
      images, label_batch = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=self.num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)

    # Display the images in the visualizer.
    tf.summary.image(self.part + '/images', images)

    return images, tf.reshape(label_batch, [batch_size])


  def _read_cifar10(self, filename_queue):
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
    self.reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = self.reader.read(filename_queue)

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
