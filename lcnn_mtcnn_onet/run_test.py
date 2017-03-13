import tensorflow as tf

import classifier.tester as tester
from model import MtcnnOnet
from dataset.celeba_reader import CelebaReader


FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument
  def create_reader():
    def simple_preprocessor(image):
      with tf.name_scope('random_simple_preprocess'):
        image = image - 0.5
        image = tf.image.central_crop(image, 0.88)  # 54 -> 48
        image = tf.reshape(image, [48, 48, 3])
      return image

    return CelebaReader(data_dir=FLAGS.celeba_data_dir,
                        batch_size=FLAGS.batch_size,
                        part=CelebaReader.DatasetPart.test,
                        processor=simple_preprocessor)


  def create_model():
    return MtcnnOnet()

  tester.test(create_model, create_reader)

if __name__ == '__main__':
  tf.app.run()
