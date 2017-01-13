import tensorflow as tf

import tester
from model import Cifar10Resnet18
from reader import Cifar10Reader


FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument
  def create_reader():
    return Cifar10Reader(data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size,
            part=Cifar10Reader.DatasetPart.train,
            preprocessing=Cifar10Reader.Preprocessing.random_simple)

  def create_model():
    return Cifar10Resnet18()

  tester.test(create_model, create_reader)

if __name__ == '__main__':
  tf.app.run()
