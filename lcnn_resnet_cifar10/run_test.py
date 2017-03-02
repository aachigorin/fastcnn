import tensorflow as tf

import classifier.tester as tester
from dataset.cifar10_reader import Cifar10Reader

from model import Cifar10LCNNResnet18


FLAGS = tf.app.flags.FLAGS


def create_reader():
  def simple_preprocessor(image):
    with tf.name_scope('simple_preprocess'):
      image = (image - Cifar10Reader.MEAN) / Cifar10Reader.STD
    return image

  return Cifar10Reader(data_dir=FLAGS.data_dir,
          batch_size=FLAGS.batch_size,
          part=Cifar10Reader.DatasetPart.test,
          processor=simple_preprocessor)


def create_model():
  return Cifar10LCNNResnet18()


def main(argv=None):  # pylint: disable=unused-argument
  tester.test(create_model, create_reader)


if __name__ == '__main__':
  tf.app.run()
