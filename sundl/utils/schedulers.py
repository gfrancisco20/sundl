"""
Tensorflow learning rate schedulers
"""

import tensorflow as tf

__all__ = ['LRScheduleDivbySteps']

class LRScheduleDivbySteps(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, initial_learning_rate, num_step = 3 , divider = 2):
    self.initial_learning_rate = initial_learning_rate
    self.num_step = num_step
    self.divider =  divider

  def __call__(self, step):
    # if (step+1) % self.num_step == 0:
    #   return self.initial_learning_rate / self.divider
    # else:
    #   return self.initial_learning_rate
    lr = self.initial_learning_rate
    newLr = self.initial_learning_rate / self.divider
    getNew = tf.cast(tf.math.equal(tf.cast((step+1) % self.num_step, dtype='float32'), 0.0), dtype='float32')
    return (1-getNew) * lr + getNew * newLr


  def get_config(self):
    config = {
        'initial_learning_rate': self.initial_learning_rate,
        'num_step': self.num_step,
        'divider': self.divider
     }
    return config