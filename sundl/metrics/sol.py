import tensorflow_probability as tfp
import tensorflow as tf

class ScoreOrientedLoss(tf.keras.losses.Loss):

  def __init__(self, name=None, reduction='sum_over_batch_size', dtype=None, tfpDistrib = None, classId = 1):
    super().__init__(name=name, reduction=reduction)
    self.tfpDistrib = tfpDistrib
    self.classId = 1
    if tfpDistrib is None:
      self.tfpDistrib = tfp.distributions.Uniform()
      
  def getY(self, y_true, y_pred):
    # if y_true.ndim > 1:
    #   if y_true.shape[1] > 1:
    if self.classId is not None:
      y_true = y_true[:,self.classId]
      y_pred = y_pred[:,self.classId]
    return y_true, y_pred
  
  def notnull(self,x):
    # isNull = tf.math.reduce_sum(tf.cast(tf.math.equal(x,0.0),self.prec))
    isNull = tf.cast(tf.math.equal(x,tf.cast(0.0,dtype =x.dtype)),dtype =x.dtype)
    epsilon = tf.experimental.numpy.finfo(x.dtype).tiny
    return isNull * epsilon + (tf.cast(1.0,dtype =isNull.dtype)-isNull) * x
  
      
  def tp(self, y_true, y_pred):
    # print(self.tfpDistrib.cdf(value=y_pred))
    # print(y_true)
    return  tf.math.reduce_sum(y_true *  tf.cast(self.tfpDistrib.cdf(value=tf.cast(y_pred,dtype='float32')),dtype=y_pred.dtype))
    
  def tn(self, y_true, y_pred):
    return  tf.math.reduce_sum((1  - y_true) * (1  - tf.cast(self.tfpDistrib.cdf(value=tf.cast(y_pred,dtype='float32')),dtype=y_pred.dtype)))
    
  def fp(self, y_true, y_pred):
    return  tf.math.reduce_sum((1 - y_true) *  tf.cast(self.tfpDistrib.cdf(value=tf.cast(y_pred,dtype='float32')),dtype=y_pred.dtype))
    
  def fn(self, y_true, y_pred):
    return  tf.math.reduce_sum(y_true *  (1 - tf.cast(self.tfpDistrib.cdf(value=tf.cast(y_pred,dtype='float32')),dtype=y_pred.dtype)))
  
  def call(self, y_true, y_pred):
    pass
  
class SOL_MCC(ScoreOrientedLoss):
  
  def __init__(self, name=None, reduction='sum_over_batch_size', dtype=None, tfpDistrib = None):
    super().__init__(name, reduction, dtype, tfpDistrib)
    
  def call(self, y_true, y_pred):
    y_true, y_pred = self.getY(y_true, y_pred)
    # print(y_true.shape, y_pred.shape)
    tp =  self.tp(y_true, y_pred)
    tn =  self.tn(y_true, y_pred)
    fp =  self.fp(y_true, y_pred)
    fn =  self.fn(y_true, y_pred)
    return 1 - ((tp*tn - fp*fn) / self.notnull(tf.math.sqrt((tp+fn)*(fn+tn) * (tn+fp)*(tp+fp))))