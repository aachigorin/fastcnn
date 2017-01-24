from __future__ import print_function

import image_preprocessing as imagenet
from imagenet_data import ImagenetData

from fastcnn.classifier.reader import BaseReader


class Imagenet2012Reader(BaseReader):
  def __init__(self, split, batch_size):
    self.dataset = ImagenetData(split)
    if split == 'train':
      self.batch = imagenet.distorted_inputs(self.dataset, batch_size=batch_size)
    elif split == 'validation':
      self.batch = imagenet.inputs(self.dataset, batch_size=batch_size)
    else:
      raise Exception('Unknown split {}'.format(split))


  def get_batch(self):
    return self.batch
