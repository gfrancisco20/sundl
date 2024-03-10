import tensorflow as tf
from sundl.explainability import getDistributedModel


def nnMcDropout(model, dropout_rate):
    '''
    Create a tf model for Monte-Carlo dropout at prediction
    model : tf model
    dropout_rate : dropout rate to apply to dropout layers
    
    Returns
    modelWithDrops : model for MC-pred ; do `modelWithDrops(x, training=True)` for preds with dropout
    '''
    
    conf = model.get_config()
    # Set dropout rate and freeze all layers but rate
    for layer in conf['layers']:
      if layer["class_name"]=="Dropout":
        layer["config"]["rate"] = dropout_rate
        layer["config"]["trainable"] = True
      elif "dropout" in layer["config"].keys():
        # print('Drop-wth-layer')
        layer["config"]["dropout"] = dropout_rate
        layer["config"]["trainable"] = True
      else:
        if "trainable" in layer["config"].keys():
          layer["config"]["trainable"] = False

    # Create the model for Mc dropout
    if type(model)==tf.keras.models.Sequential:
      model_dropout = tf.keras.models.Sequential.from_config(conf)
    else:
      model_dropout = tf.keras.models.Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    modelWithDrops = tf.keras.models.Model(model_dropout.inputs, model_dropout.outputs, name='nnMcDropout')
    
    return modelWithDrops
  
def pcnnMcDropout(model, dropout_rate, innerCnnLayerName ='time_distributed', num_ptc = 8):
  '''
    Create a pcnn for Monte-Carlo dropout at prediction
    model : tf pcnn
    dropout_rate : dropout rate to apply to dropout layers
    
    Returns
    modelWithDrops : model for MC-pred ; do `modelWithDrops(x, training=True)` for preds with dropout
  '''
  innerModel = getDistributedModel(model, distributedLayerName = innerCnnLayerName)
  innerModelWithDrops = nnMcDropout(innerModel, dropout_rate)
  
  # patchEtractor = tf.keras.models.Model(model.input, model.get_layer('time_distributed').input)
  for layer in model.layers:
    # if layer.name == distributedLayerName:
    if innerCnnLayerName in layer.name:
      innerInput = layer.input
  patchEtractor = tf.keras.models.Model(model.input, innerInput)

  # patches_outputs =  tf.keras.layers.TimeDistributed(innerModelWithDrops)(patchEtractor.output)
  # patches_outputs = [patches_outputs[:,ptIdx,:] for ptIdx in range(patches_outputs.shape[1])]
  patches_outputs = []
  for ptIdx in range(num_ptc): 
    patches_outputs.append( innerModelWithDrops(patchEtractor.output[:,ptIdx]) )

  output = model.layers[-1](patches_outputs)
  modelWithDrops = tf.keras.Model(model.input, output, name="pcnnMcDropout")
  return modelWithDrops
