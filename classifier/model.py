from abc import ABCMeta, abstractmethod


class BaseModel(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def inference(self, images, is_train):
    pass


  @abstractmethod
  def loss(self, logits, labels):
    pass
