import tensorflow as tf

def zero_const():
  return tf.constant(0, shape=[1])


def minus_one_const():
  return tf.constant(-1, shape=[1])


def count(t, val):
  elements_equal_to_value = tf.equal(t, val)
  as_ints = tf.cast(elements_equal_to_value, tf.int32)
  count = tf.reduce_sum(as_ints, keep_dims=True)
  return count


def stack_and_squeeze(dims):
  return tf.squeeze(tf.stack(dims, axis=0))