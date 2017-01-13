from abc import ABCMeta, abstractmethod

class BaseReader(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_batch():
    pass
