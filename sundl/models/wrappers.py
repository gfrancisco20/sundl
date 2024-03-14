"""
Tensorflow models utilitary tools
"""

from functools import reduce
import dill as pickle
import tensorflow as tf

from sundl.utils.schedulers import LRScheduleDivbySteps

__all__ = ['ModelInstantier',
           'reinstatiateOptim'
           ]

class ModelInstantier():
  """
  Class wrapping model and dataset building functions and parameters
  along with properties useful in training loops
  It's purpose is to ease hyperparameter search on combinations of
  dataset and models parameters

  Attributes
  ----------
  buildModelFunction : func
    function that instantiate a tensorflow model
  buildModelParams : dict
    params of buildModelFunction
  buildDsFunction : func
    function that instantiate a tensorflow datasset
  buildDsParams : dict
    params of buildDsFunction
  name : str
    general name of the model
  savedPredictionModel : bool
    True for constant model (input=output)
  cls : str
    tag for binary classifier
  config : dict
    store datasets (keys : 'train','val','test') 
    and models (key : 'model') parameters , 
    updated when models (__call__) and datasets are instantiated (build_DS)
    
    
  Methods
  ----------
  __call__() -> tfModel
    instantiate a tfModel with buildModelFunction(**buildModelParams),
    updates config['model']
    
  build_DS(self, **kwargs) -> tf.data.Dataset
    instantiate a tf.Dataset with buildDsFunction(**buildDsParams, **kwargs),
    updates config[kwargs['typeDS']] if 'typeDS' in kwargs.keys()
    
  saveConfig(pathConfig)
    save config to pathConfig,
    usefull for reinstatiating saved models and 
    recovering metrics, training and dataset parameters
  
  fullNameFunc(name, *args) -> str
    return a detailed model name based on 'name' 
    and dataset characteristics 'args'

  """

  def __init__(self,
    buildModelFunction,
    buildModelParams,
    buildDsFunction,
    buildDsParams,
    name,
    # fullNameFunc=None,
    savedPredictionModel = False,
    cls = None
    ):
    """
    Parameters
    ----------
    buildModelFunction : func
      function that instantiate a tensorflow model
    buildModelParams : dict
      params of buildModelFunction
    buildDsFunction : func
      function that instantiate a tensorflow datasset
    buildDsParams : dict
      params of buildDsFunction
    name : str
      general name of the model
    savedPredictionModel : bool, optional
      True for constant model (input=output), default to False 
    cls : str, optional
      tag for binary classifier
    fullNameFunc : func, optional
      function generating detailed model name based on self.name and dataset characteristics
    """
    self.cls = cls
    self.buildModelFunction = buildModelFunction
    self.buildModelParams = buildModelParams
    self.buildDsFunction = buildDsFunction
    self.buildDsParams = buildDsParams
    self.name = name
    self.savedPredictionModel = savedPredictionModel
    self.config = {}
    # if fullNameFunc is None:
    #   self.fullNameFunc = lambda name, timesteps, channels, h: name + '_' +reduce(lambda x,y:x+'x'+y,[f'{channel:0>4}' for channel in channels]) + f'_{h}'
    # else:
    #   self.fullNameFunc = fullNameFunc
    
  # overwrite by inherating ModelInstantier for
  # custum function names based on model properties/parameters
  def fullNameFunc(self, channels, h):
    return  self.name + '_' +reduce(lambda x,y:x+'x'+y,[f'{channel:0>4}' for channel in channels]) + f'_{h}'

  def __call__(self):
    ''' return a TF model'''
    self.config['model'] = self.buildModelParams
    return self.buildModelFunction(**self.buildModelParams)

  def build_DS(self, **kwargs):
    params2rmvIfNone = ['pathDir', 'channels', 'timesteps']
    for pname in params2rmvIfNone:
      if pname in list(kwargs.keys()):
        if kwargs[pname] is None: 
          kwargs.pop(pname)
    self.buildDsParams.update(kwargs)
    # typeDs in ['train','val','test']:
    if 'typeDs' in self.buildDsParams.keys():
      typeDs = self.buildDsParams['typeDs']
      self.buildDsParams.pop('typeDs')
      self.config[f'dataset_{typeDs}'] = self.buildDsParams
      self.config[f'buildDsFunction'] = self.buildDsFunction
    return self.buildDsFunction(**self.buildDsParams)
  
  def saveConfig(self, pathConfig):
    tmp = self.config.copy()
    savedConfig = {}
    for k in tmp.keys():
      if isinstance(tmp[k], dict):
        savedConfig[k] = tmp[k].copy()
      else:
        savedConfig[k] = tmp[k]
    try:
      with open(pathConfig, 'wb') as f1:
        pickle.dump(savedConfig, f1)
    except Exception as e:
      print(e)
      print('Saving config without tfModel and tf-layers')
      for modelParam in self.config['model']:
        if modelParam == 'tfModel':
          savedConfig['model']['tfModel'] = savedConfig['model']['tfModel'].__name__
        if isinstance(self.config['model'][modelParam], tf.keras.layers.Layer):
          savedConfig['model'][modelParam] = savedConfig['model'][modelParam].name
      with open(pathConfig, 'wb') as f1:
        pickle.dump(savedConfig, f1)
    
def reinstatiateOptim(optimizer):
  
  if type(optimizer)!=str:
    # optiConfig = optimizer.get_config()
    # if type(optiConfig['learning_rate']) == dict:
    #   if optiConfig['learning_rate']['class_name'] == 'LRScheduleDivbySteps':
    #     optiConfig['learning_rate'] = LRScheduleDivbySteps(**optiConfig['learning_rate']['config'])
    # if optiConfig['name'] == 'Adam':
    #   optimizer = tf.keras.optimizers.Adam(
    #           learning_rate= optiConfig['learning_rate'],
    #           weight_decay=optiConfig['weight_decay']
    #         )
    # if optiConfig['name'] == 'AdamW':
    #   optimizer = tf.keras.optimizers.AdamW(
    #           learning_rate= optiConfig['learning_rate'],
    #           weight_decay=optiConfig['weight_decay']
    # )
    optimizer = optimizer.from_config(optimizer.get_config())
  return optimizer
