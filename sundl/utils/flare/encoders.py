"""
mpf and toteh encoders and decoders for regression on those variables
"""

import tensorflow as tf

from sundl.utils.flare.thresholds import mpfTresh, totehTresh

__all__ = ['mpfEncoder',
           'mpfDecoder',
           'mpf_C_metrics_decoder',
           'mpf_M_metrics_decoder',
           'mpf_X_metrics_decoder',
           'totehEncoders',
           'totehDecoders',
           'toteh_C_metrics_decoders',
           'toteh_M_metrics_decoders',
           'toteh_X_metrics_decoders'
           
          ]

def mpfEncoder(flux):
  """
  log-linearisation of the mpf in a 0-12 range
  """
  base = tf.math.log(10.0)
  base = tf.cast(base, dtype=tf.float32)
  exp =  1-tf.math.log(mpfTresh['quiet'][1]) / base
  res = tf.math.log(1 + tf.math.pow(10.0,exp)*flux)
  return res

def mpfDecoder(flux):
  """
  retrieve original mpf values from mpfEncoder outputs
  """
  base = tf.math.log(10.0)
  base = tf.cast(base, dtype=tf.float32)
  exp =  1-tf.math.log(mpfTresh['quiet'][1]) / base
  res =  (tf.math.exp(flux) - 1) / tf.math.pow(10.0,exp)
  return res

def mpf_C_metrics_decoder(flux):
  """
  convert flux encoded by mpfEncoder into C+ binary value,
  allow to compute C+ metrics on a mpfEncoder-eencoded regression model 
  """
  return tf.cast(mpfDecoder(flux)>=mpfTresh['C'][0], dtype = 'float32')

def mpf_M_metrics_decoder(flux):
  """
  convert flux encoded by mpfEncoder into M+ binary value,
  allow to compute M+ metrics on a mpfEncoder-eencoded regression model 
  """
  return tf.cast(mpfDecoder(flux)>=mpfTresh['M'][0], dtype = 'float32')

def mpf_X_metrics_decoder(flux):
  """
  convert flux encoded by mpfEncoder into X+ binary value,
  allow to compute X+ metrics on a mpfEncoder-eencoded regression model 
  """
  return tf.cast(mpfDecoder(flux)>=mpfTresh['X'][0], dtype = 'float32')

""" log-linearisation of the toteh in a 0-12 range """
totehEncoders = {}
""" retrieve original mpf values from totehEncoders outputs """
totehDecoders = {}
""" convert flux encoded by totehEncoders into C+ binary value, """
toteh_C_metrics_decoders = {}
""" convert flux encoded by totehEncoders into M+ binary value, """
toteh_M_metrics_decoders = {}
""" convert flux encoded by totehEncoders into X+ binary value, """
toteh_X_metrics_decoders = {}

# for window_h in totehTresh.keys():
#   totehEncoders[window_h] = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[window_h]['quiet'][1]) / tf.math.log(10.0) )*flux)
#   totehDecoders[window_h] = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[window_h]['quiet'][1]) / tf.math.log(10.0))
#   toteh_C_metrics_decoders[window_h] = lambda flux : tf.cast(totehDecoders[window_h](flux)>=totehTresh[window_h]['C'][0], dtype = 'float32')
#   toteh_M_metrics_decoders[window_h] = lambda flux : tf.cast(totehDecoders[window_h](flux)>=totehTresh[window_h]['M'][0], dtype = 'float32')
#   toteh_X_metrics_decoders[window_h] = lambda flux : tf.cast(totehDecoders[window_h](flux)>=totehTresh[window_h]['X'][0], dtype = 'float32')

# !!!! Loop def not working --> right side window_h = last totehTresh.keys() for all iteration after import ?!

totehEncoders[2]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[2]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[2]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[2]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[2] = lambda flux : tf.cast(totehDecoders[2](flux)>=totehTresh[2]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[2] = lambda flux : tf.cast(totehDecoders[2](flux)>=totehTresh[2]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[2] = lambda flux : tf.cast(totehDecoders[2](flux)>=totehTresh[2]['X'][0], dtype = 'float32')

totehEncoders[4]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[4]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[4]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[4]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[4] = lambda flux : tf.cast(totehDecoders[4](flux)>=totehTresh[4]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[4] = lambda flux : tf.cast(totehDecoders[4](flux)>=totehTresh[4]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[4] = lambda flux : tf.cast(totehDecoders[4](flux)>=totehTresh[4]['X'][0], dtype = 'float32')

totehEncoders[8]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[8]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[8]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[8]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[8] = lambda flux : tf.cast(totehDecoders[8](flux)>=totehTresh[8]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[8] = lambda flux : tf.cast(totehDecoders[8](flux)>=totehTresh[8]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[8] = lambda flux : tf.cast(totehDecoders[8](flux)>=totehTresh[8]['X'][0], dtype = 'float32')

totehEncoders[12]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[12]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[12]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[12]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[12] = lambda flux : tf.cast(totehDecoders[12](flux)>=totehTresh[12]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[12] = lambda flux : tf.cast(totehDecoders[12](flux)>=totehTresh[12]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[12] = lambda flux : tf.cast(totehDecoders[12](flux)>=totehTresh[12]['X'][0], dtype = 'float32')

totehEncoders[24]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[24]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[24]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[24]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[24] = lambda flux : tf.cast(totehDecoders[24](flux)>=totehTresh[24]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[24] = lambda flux : tf.cast(totehDecoders[24](flux)>=totehTresh[24]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[24] = lambda flux : tf.cast(totehDecoders[24](flux)>=totehTresh[24]['X'][0], dtype = 'float32')

totehEncoders[36]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[36]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[36]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[36]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[36] = lambda flux : tf.cast(totehDecoders[36](flux)>=totehTresh[36]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[36] = lambda flux : tf.cast(totehDecoders[36](flux)>=totehTresh[36]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[36] = lambda flux : tf.cast(totehDecoders[36](flux)>=totehTresh[36]['X'][0], dtype = 'float32')

totehEncoders[48]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[48]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[48]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[48]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[48] = lambda flux : tf.cast(totehDecoders[48](flux)>=totehTresh[48]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[48] = lambda flux : tf.cast(totehDecoders[48](flux)>=totehTresh[48]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[48] = lambda flux : tf.cast(totehDecoders[48](flux)>=totehTresh[48]['X'][0], dtype = 'float32')

totehEncoders[72]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[72]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[72]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[72]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[72] = lambda flux : tf.cast(totehDecoders[72](flux)>=totehTresh[72]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[72] = lambda flux : tf.cast(totehDecoders[72](flux)>=totehTresh[72]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[72] = lambda flux : tf.cast(totehDecoders[72](flux)>=totehTresh[72]['X'][0], dtype = 'float32')

totehEncoders[144]            = lambda flux : tf.math.log(1 + tf.math.pow(10.0,1-tf.math.log(totehTresh[144]['quiet'][1]) / tf.math.log(10.0) )*flux)
totehDecoders[144]            = lambda flux : (tf.math.exp(flux) - 1) / tf.math.pow(10.0,1-tf.math.log(totehTresh[144]['quiet'][1]) / tf.math.log(10.0))
toteh_C_metrics_decoders[144] = lambda flux : tf.cast(totehDecoders[72](flux)>=totehTresh[144]['C'][0], dtype = 'float32')
toteh_M_metrics_decoders[144] = lambda flux : tf.cast(totehDecoders[72](flux)>=totehTresh[144]['M'][0], dtype = 'float32')
toteh_X_metrics_decoders[144] = lambda flux : tf.cast(totehDecoders[72](flux)>=totehTresh[144]['X'][0], dtype = 'float32')


