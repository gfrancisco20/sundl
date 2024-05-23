"""
Classification and regression metrics for tensorflow
"""
from abc import ABC, abstractmethod

import tensorflow as tf

# __all__ = ['MAE',
#            'RMSE',
#            'Recall',
#            'Precision',
#            'F1',
#            'Tss',
#            'Mcc',
#            'HssAlt',
#            'Far',
#            'TP',
#            'TN',
#            'FP',
#            'FN'
#            ]

epsilon = 1e-20

class MultiClassMetrics(tf.keras.metrics.Metric):#,ABC):
  def __init__(self, 
               name, 
               threshold = 0.5, 
               classDecoders = None, 
               num_class = 3, 
               **kwargs
  ):
    super(MultiClassMetrics, self).__init__(name=name, **kwargs)
    self.tp = [self.add_weight(name=f'tp_{clsIdx}', initializer='zeros') for clsIdx in range(num_class)]
    self.tn = [self.add_weight(name=f'tn_{clsIdx}', initializer='zeros') for clsIdx in range(num_class)]
    self.fp = [self.add_weight(name=f'fp_{clsIdx}', initializer='zeros') for clsIdx in range(num_class)]
    self.fn = [self.add_weight(name=f'fn_{clsIdx}', initializer='zeros') for clsIdx in range(num_class)]
    self.threshold = threshold
    self.num_class = num_class
    
    if threshold > 1: 
      threshold = threshold/100
    if threshold == 0.5:
        self.__name__ = name
    else:
        self.__name__ = name+f'_{100*threshold:0>2.0f}'
 
    if classDecoders is None:
      classDecoders = []
      for clsIdx in range(self.num_class):
        classDecoders[clsIdx] = lambda y_true, y_pred: (tf.cast(y_true[:,clsIdx]>self.threshold, dtype=tf.float32), 
                                                        tf.cast(y_pred[:,clsIdx]>self.threshold, dtype=tf.float32))
    self.classDecoders = classDecoders
    
  def __true_positives(self, y_transform, y_true, y_pred):
    y_true, y_pred = y_transform(y_true, y_pred)
    return tf.reduce_sum(y_true * y_pred) #tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
  def __true_negatives(self, y_transform, y_true, y_pred,):
    y_true, y_pred = y_transform(y_true, y_pred)
    return tf.reduce_sum((y_true-1) * (y_pred-1)) #tf.reduce_sum(tf.round(tf.clip_by_value((y_true-1) * (y_pred-1), 0, 1)))
  def __false_positives(self, y_transform, y_true, y_pred):
    y_true, y_pred = y_transform(y_true, y_pred)
    return tf.reduce_sum(-(y_true-1) * (y_pred)) #tf.reduce_sum(tf.round(tf.clip_by_value(-(y_true-1) * (y_pred), 0, 1)))
  def __false_negatives(self, y_transform, y_true, y_pred):
    y_true, y_pred = y_transform(y_true, y_pred)
    return tf.reduce_sum(-(y_true) * (y_pred-1)) #tf.reduce_sum(tf.round(tf.clip_by_value(-(y_true) * (y_pred-1), 0, 1)))

  def update_state(self, y_true, y_pred, sample_weight=None):
    for clsIdx in range(self.num_class):
      y_transform = lambda x,y: (self.classDecoders[clsIdx](x), self.classDecoders[clsIdx](y))
      tp = self.__true_positives(y_transform, y_true, y_pred)
      tn = self.__true_negatives(y_transform, y_true, y_pred)
      fp = self.__false_positives(y_transform, y_true, y_pred)
      fn = self.__false_negatives(y_transform, y_true, y_pred)
      self.tp[clsIdx].assign_add(tf.reduce_sum(tp))
      self.tn[clsIdx].assign_add(tf.reduce_sum(tn))
      self.fp[clsIdx].assign_add(tf.reduce_sum(fp))
      self.fn[clsIdx].assign_add(tf.reduce_sum(fn))
    
  def all_positives(self, clsIdx):
    return self.tp[clsIdx] + self.fn[clsIdx]
  def all_negatives(self, clsIdx):
    return self.tn[clsIdx] + self.fp[clsIdx]
  def predicted_positives(self, clsIdx):
    return self.tp[clsIdx] + self.fp[clsIdx]
  def predicted_negatives(self, clsIdx):
    return self.tn[clsIdx] + self.fn[clsIdx]

  @abstractmethod
  def result(self):
    pass
  
class Mcc_Multi(MultiClassMetrics):
  def __init__(self, threshold = 0.5, name = None, classDecoders = None, num_class = 3, averaging = 'macro', macroWeights = None):
    if name is None: name = 'mcc'
    super().__init__(name, threshold = threshold, classDecoders = classDecoders, num_class = num_class)
    self.averaging = averaging
    if macroWeights is None:
      macroWeights = [1 for k in range(num_class)]
    self.macroWeights = macroWeights
  def formula(self, tp, tn, fp, fn):
    num = tp*tn - fp*fn
    denom =  (tp+fn) * (fn+tn) * (tn+fp) * (tp+fp)
    return num/notnull(tf.math.sqrt(denom))
  def result(self):
    if self.averaging == 'macro':
      mcc = 0
      totWeight = 0
      for clsIdx in range(self.num_class):
        tp = self.tp[clsIdx]
        tn = self.tn[clsIdx]
        fp = self.fp[clsIdx]
        fn = self.fn[clsIdx]
        mcc += self.macroWeights[clsIdx] * self.formula(tp, tn, fp, fn)
        totWeight += self.macroWeights[clsIdx]
      mcc /= totWeight
    elif self.averaging == 'micro':
      tp = 0
      tn = 0
      fp = 0
      fn = 0
      for clsIdx in range(self.num_class):
        tp += self.tp[clsIdx]
        tn += self.tn[clsIdx]
        fp += self.fp[clsIdx]
        fn += self.fn[clsIdx]
      mcc = self.formula(tp, tn, fp, fn)
    return mcc
      
    
      


class BinaryMetrics(tf.keras.metrics.Metric):#,ABC):
  def __init__(self, 
               name, 
               threshold = 0.5, 
               y_transform = 'one_hot', 
               labelDecoder = None, 
               classId = 1, 
               **kwargs
  ):
    super(BinaryMetrics, self).__init__(name=name, **kwargs)
    self.tp = self.add_weight(name='tp', initializer='zeros')
    self.tn = self.add_weight(name='tn', initializer='zeros')
    self.fp = self.add_weight(name='fp', initializer='zeros')
    self.fn = self.add_weight(name='fn', initializer='zeros')
    self.threshold = threshold
    self.classId = classId
    
    self.available_y_transforms = ['one_hot', 'binary_linear']
    if threshold > 1: 
      threshold = threshold/100
    if threshold == 0.5:
        self.__name__ = name
    else:
        self.__name__ = name+f'_{100*threshold:0>2.0f}'
 
    self.labelDecoder = labelDecoder
    if type(y_transform)==str:
        if y_transform not in self.available_y_transforms:
            raise Exception(f'Choose y_transform in {self.available_y_transforms } or pass your own y_transform function in argument')
        elif y_transform=='one_hot':
            self.y_transform = self.__one_cold_decode
        elif y_transform=='binary_linear':
            self.y_transform = self.__binary_linear
    else:
      self.y_transform = y_transform
     # mapping if neccessary for __flux2srxClasses's f2c values (nn output might be a function of the actual flux, labelDecoder should be the inverse of that function)
    # TEMPORARY
    if y_transform is None and labelDecoder is not None:
      def y_transform(y_true, y_pred=None):
        if y_pred is None:
          return self.labelDecoder(y_true)
        else:
          return self.labelDecoder(y_true), self.labelDecoder(y_pred)
      self.y_transform = y_transform

  def __binary_linear(self, y_true, y_pred=None):
      # TODOs
      # ??
      pass
  def __one_cold_decode(self, y_true, y_pred=None):
    if self.classId is None:
      decoder = lambda y: tf.argmax(y,axis=1)
    else:
      # only binary case
      decoder = lambda y: (y[:,self.classId]>self.threshold)#[:]
    if y_pred is None:
      return  tf.cast(decoder(y_true), dtype=tf.float32)
    else:
      return tf.cast(decoder(y_true), dtype=tf.float32) , tf.cast(decoder(y_pred), dtype=tf.float32)

  def __true_positives(self, y_true, y_pred):
    y_true, y_pred = self.y_transform(y_true, y_pred)
    return tf.reduce_sum(y_true * y_pred) #tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
  def __true_negatives(self, y_true, y_pred,):
    y_true, y_pred = self.y_transform(y_true, y_pred)
    return tf.reduce_sum((y_true-1) * (y_pred-1)) #tf.reduce_sum(tf.round(tf.clip_by_value((y_true-1) * (y_pred-1), 0, 1)))
  def __false_positives(self, y_true, y_pred):
    y_true, y_pred = self.y_transform(y_true, y_pred)
    return tf.reduce_sum(-(y_true-1) * (y_pred)) #tf.reduce_sum(tf.round(tf.clip_by_value(-(y_true-1) * (y_pred), 0, 1)))
  def __false_negatives(self, y_true, y_pred):
    y_true, y_pred = self.y_transform(y_true, y_pred)
    return tf.reduce_sum(-(y_true) * (y_pred-1)) #tf.reduce_sum(tf.round(tf.clip_by_value(-(y_true) * (y_pred-1), 0, 1)))

  def update_state(self, y_true, y_pred, sample_weight=None):
    tp = self.__true_positives(y_true, y_pred)
    tn = self.__true_negatives(y_true, y_pred)
    fp = self.__false_positives(y_true, y_pred)
    fn = self.__false_negatives(y_true, y_pred)
    self.tp.assign_add(tf.reduce_sum(tp))
    self.tn.assign_add(tf.reduce_sum(tn))
    self.fp.assign_add(tf.reduce_sum(fp))
    self.fn.assign_add(tf.reduce_sum(fn))
    
  def all_positives(self):
    return self.tp + self.fn
  def all_negatives(self):
    return self.tn + self.fp
  def predicted_positives(self):
    return self.tp + self.fp
  def predicted_negatives(self):
    return self.tn + self.fn

  @abstractmethod
  def result(self):
    pass

class RegressionMetrics(tf.keras.metrics.Metric):#,ABC):
  def __init__(self, 
               name, 
               threshold = 0.5, 
               y_transform = 'one_hot', 
               labelDecoder = None, 
               classId = None, 
               **kwargs
  ):
    super(RegressionMetrics, self).__init__(name=name, **kwargs)
    self.threshold = threshold
    self.classId = classId
    
    if threshold > 1: 
      threshold = threshold/100
    if threshold == 0.5:
        self.__name__ = name
    else:
        self.__name__ = name+f'_{100*threshold:0>2.0f}'
 
    self.labelDecoder = labelDecoder

    self.y_transform = y_transform
     # mapping if neccessary for __flux2srxClasses's f2c values (nn output might be a function of the actual flux, labelDecoder should be the inverse of that function)
    # TEMPORARY
    if y_transform is None and labelDecoder is not None:
      def y_transform(y_true, y_pred=None):
        if y_pred is None:
          return self.labelDecoder(y_true)
        else:
          return self.labelDecoder(y_true), self.labelDecoder(y_pred)
      self.y_transform = y_transform


  @abstractmethod
  def update_state(self):
    pass

  @abstractmethod
  def result(self):
    pass
  
  # def __call__(self, y_true, y_pred=None):
  #   return self.compute_metric(y_true, y_pred)

def notnull(x):
  # isNull = tf.math.reduce_sum(tf.cast(tf.math.equal(x,0.0),prec))
  isNull = tf.cast(tf.math.equal(x,0.0),prec)
  return isNull * epsilon + (1.0-isNull) * x
  # if x==0:
  #   return epsilon # tf.keras.backend.epsilon()
  # else:
  #   return x
  
class MAE_weighted(RegressionMetrics):
    def __init__(self, y_transform=None, name=None, labelDecoder=None, classId=None):
        if name is None:
            name = 'MAE_weighted'
        super().__init__(name, y_transform=y_transform, labelDecoder=labelDecoder, classId=classId)
        self.total_abs_error = self.add_weight(name='total_abs_error', initializer='zeros')
        self.total_weight = self.add_weight(name='total_weight', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.labelDecoder is not None:
            y_true = self.labelDecoder(y_true)
            y_pred = self.labelDecoder(y_pred)
        
        if self.classId is None:
            absolute_errors = tf.abs(y_true - y_pred)
        else:
            absolute_errors = tf.abs(y_true[:, self.classId] - y_pred[:, self.classId])
        
        if sample_weight is None:
            sample_weight = tf.ones_like(absolute_errors)

        weighted_abs_error = tf.reduce_sum(absolute_errors * sample_weight)
        total_weight = tf.reduce_sum(sample_weight)

        self.total_abs_error.assign_add(weighted_abs_error)
        self.total_weight.assign_add(total_weight)

    def result(self):
        return self.total_abs_error / self.total_weight
      
class MAPE_weighted(RegressionMetrics):
    def __init__(self, y_transform=None, name=None, labelDecoder=None, classId=None):
        if name is None:
            name = 'MAPE_weighted'
        super().__init__(name, y_transform=y_transform, labelDecoder=labelDecoder, classId=classId)
        self.weighted_errors = self.add_weight(name='weighted_errors', initializer='zeros')
        self.total_weight = self.add_weight(name='total_weight', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.labelDecoder is not None:
            y_true = self.labelDecoder(y_true)
            y_pred = self.labelDecoder(y_pred)
        
        if self.classId is None:
            absolute_errors = tf.abs((y_true - y_pred) / notnull(y_true))
        else:
            absolute_errors = tf.abs((y_true[:, self.classId] - y_pred[:, self.classId]) / notnull(y_true[:, self.classId]))
        
        if sample_weight is None:
            sample_weight = tf.ones_like(absolute_errors)

        weighted_abs_error = tf.reduce_sum(absolute_errors * sample_weight)
        total_weight = tf.reduce_sum(sample_weight)

        self.weighted_errors.assign_add(weighted_abs_error)
        self.total_weight.assign_add(total_weight)

    def result(self):
        return self.weighted_errors / self.total_weight

class MAE(RegressionMetrics):
  def __init__(self, y_transform = None, name = None, labelDecoder = None,classId = None):
    if name is None: name = 'MAE'
    super().__init__(name, y_transform = y_transform, labelDecoder = labelDecoder, classId=classId)
    self.errs = self.add_weight(name='errs', initializer='zeros')
    self.size = self.add_weight(name='size', initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.labelDecoder is not None:
      y_true = self.labelDecoder(y_true)
      y_pred = self.labelDecoder(y_pred)
    if self.classId is None:
      errs = tf.reduce_sum(tf.abs(y_true-y_pred))
    else:
      errs = tf.reduce_sum(tf.abs(y_true[:,self.classId]-y_pred[:,self.classId]))
    # size = tf.cast(y_pred.shape[0],prec)
    if self.classId is not None:
      size = tf.cast(len(y_pred),prec)
    else:
      size = tf.cast(y_pred.shape[0]*y_pred.shape[1],prec)
    self.errs.assign_add(errs) 
    self.size.assign_add(size)    
  
  def result(self):
      return self.errs / self.size
      

class MAPE(RegressionMetrics):
  def __init__(self, y_transform = None, name = None, labelDecoder = None,classId = None):
    if name is None: name = 'MAPE'
    super().__init__(name, y_transform = y_transform, labelDecoder = labelDecoder, classId=classId)
    self.errs = self.add_weight(name='errs', initializer='zeros')
    self.size = self.add_weight(name='size', initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.labelDecoder is not None:
      y_true = self.labelDecoder(y_true)
      y_pred = self.labelDecoder(y_pred)
    if self.classId is None:
      errs = tf.reduce_sum(tf.abs((y_true-y_pred)/notnull(y_true))) 
    else:
      errs = tf.reduce_sum(tf.abs((y_true[:,self.classId]-y_pred[:,self.classId])/notnull(y_true[:,self.classId])))
    # size = tf.cast(y_pred.shape[0],prec)
    if self.classId is not None:
      size = tf.cast(len(y_pred),prec)
    else:
      size = tf.cast(y_pred.shape[0]*y_pred.shape[1],prec)
    self.errs.assign_add(errs) 
    self.size.assign_add(size)    
  
  def result(self):
      return self.errs / self.size

class RMSE(RegressionMetrics):
  def __init__(self, y_transform = None, name = None, labelDecoder = None, classId = None):
    if name is None: name = 'RMSE'
    super().__init__(name, y_transform = y_transform, labelDecoder = labelDecoder, classId=classId)
    self.errs = self.add_weight(name='errs', initializer='zeros')
    self.size = self.add_weight(name='size', initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.labelDecoder is not None:
      y_true = self.labelDecoder(y_true)
      y_pred = self.labelDecoder(y_pred)
    if self.classId is None:
      errs = tf.reduce_sum(tf.square(y_true-y_pred))
    else:
      errs = tf.reduce_sum(tf.square(y_true[:,self.classId]-y_pred[:,self.classId]))
    # size = tf.cast(y_pred.shape[0],prec)
    if self.classId is not None:
      size = tf.cast(len(y_pred),prec)
    else:
      size = tf.cast(y_pred.shape[0]*y_pred.shape[1],prec)
    self.errs.assign_add(errs) 
    self.size.assign_add(size)    
  
  def result(self):
    return tf.sqrt(self.errs / self.size)
  
class R2(RegressionMetrics):
  def __init__(self, y_transform = None, name = None, labelDecoder = None, classId = None):
    if name is None: name = 'r2'
    super().__init__(name, y_transform = y_transform, labelDecoder = labelDecoder, classId=classId)
    self.size   = self.add_weight(name='size', initializer='zeros')
    self.yTrue  = self.add_weight(name='yTrue', initializer='zeros')
    self.yTrue2 = self.add_weight(name='yTrue2', initializer='zeros')
    self.yPred  = self.add_weight(name='yPred', initializer='zeros')
    self.yPred2 = self.add_weight(name='yPred2', initializer='zeros')
    self.prod   = self.add_weight(name='prod', initializer='zeros')
    
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.labelDecoder is not None:
      y_true = self.labelDecoder(y_true)
      y_pred = self.labelDecoder(y_pred)
    if self.classId is None:
      yTrue = tf.reduce_sum(y_true)
      yPred = tf.reduce_sum(y_pred)
      yTrue2 = tf.reduce_sum(tf.square(y_true))
      yPred2 = tf.reduce_sum(tf.square(y_pred))
      prod = tf.reduce_sum(y_true * y_pred )
    else:
      yTrue = tf.reduce_sum(y_true[:,self.classId])
      yPred = tf.reduce_sum(y_pred[:,self.classId])
      yTrue2 = tf.reduce_sum(tf.square(y_true[:,self.classId]))
      yPred2 = tf.reduce_sum(tf.square(y_pred[:,self.classId]))
      # prod = tf.reduce_sum(y_true[:,self.classId] * y_pred[:,self.classId] )
      prod = tf.reduce_sum(tf.math.multiply(y_true[:,self.classId],y_pred[:,self.classId])) 
    if self.classId is not None:
      size = tf.cast(len(y_pred),prec)
    else:
      # n = y_pred.shape[0]*y_pred.shape[1]
      n  = tf.reduce_sum(tf.cast(tf.math.equal(y_pred,y_pred),prec))
      size = tf.cast(n,prec)
    self.size.assign_add(size)   
    self.yTrue.assign_add(yTrue)   
    self.yTrue2.assign_add(yTrue2)   
    self.yPred.assign_add(yPred)   
    self.yPred2.assign_add(yPred2)   
    self.prod.assign_add(prod)    
    
  def result(self):
    n = self.size
    n2 = n*n
    return tf.square(self.prod/n - self.yTrue * self.yPred / n2) / ( (self.yTrue2/n - tf.square(self.yTrue/n)) * (self.yPred2/n - tf.square(self.yPred/n) ) )
  

class Recall(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'recall'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return self.tp / notnull(self.all_positives())

class Precision(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'precision'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return self.tp / notnull(self.predicted_positives())

class F1(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'f1'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return 2 * self.tp / notnull(2*self.tp + self.fp + self.fn)

class Tss(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'tss'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
        self.recall = Recall(threshold, y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        # tss = tpr + tnr - 1
        recall = self.tp / notnull(self.all_positives())
        tnr = self.tn / notnull(self.all_negatives())
        return recall + tnr - 1

class HssAlt(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'hss_alt'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
        self.precision = Precision(threshold, y_transform, labelDecoder = labelDecoder, classId = classId)
        self.recall = Recall(threshold, y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        recall = self.tp / notnull(self.all_positives())
        precision = self.tp / notnull(self.predicted_positives())
        return recall * (2 - 1/notnull(precision))

class Hss(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'hss'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        num = 2*(self.tp*self.tn - self.fp*self.fn)
        denom =  self.all_positives() * (self.fn+self.tn) + self.all_negatives() * (self.tp+self.fp)
        return num/notnull(denom)

class Mcc(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'mcc'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        num = self.tp*self.tn - self.fp*self.fn
        denom =  self.all_positives() * (self.fn+self.tn) * self.all_negatives() * (self.tp+self.fp)
        return num/notnull(tf.math.sqrt(denom))


class Far(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'far'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        # FP / (TP + FP)
        return self.fp / notnull(self.predicted_positives())

class TP(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'TP'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return self.tp
class TN(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'TN'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return self.tn
class FP(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'FP'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return self.fp
class FN(BinaryMetrics):
    def __init__(self, threshold = 0.5, y_transform = 'one_hot', name = None, labelDecoder = None, classId = 1):
        if name is None: name = 'FN'
        super().__init__(name, threshold = threshold, y_transform = y_transform, labelDecoder = labelDecoder, classId = classId)
    def result(self):
        return self.fn