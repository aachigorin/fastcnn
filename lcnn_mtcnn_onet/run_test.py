import tensorflow as tf

import classifier.tester as tester
from model import MtcnnOnet
from dataset.celeba_widerface_reader import CelebaWiderfaceReader


FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument
  def create_reader():
    return CelebaWiderfaceReader(batch_size=FLAGS.batch_size,
                                 part=CelebaWiderfaceReader.DatasetPart.test)


  def create_model():
    return MtcnnOnet()

  tester.test(create_model, create_reader)

if __name__ == '__main__':
  tf.app.run()
