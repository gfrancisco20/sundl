
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ['getDistributedModel',
           'preprocIm',
           'visual_explainer',
           'gradCam',
           'guidedBackProp',
           'visual_explainer',
           'EnsembleExplainabilityRGB'
           ]

def preprocIm(images, preprocFunc = tf.keras.applications.efficientnet_v2.preprocess_input):
  preprocessed_img = preprocFunc(images)
  imgArr = preprocessed_img.numpy()[0].astype('uint8')
  return images, imgArr, preprocessed_img

def getDistributedModel(model, distributedLayerName = 'time_distributed',custom_objects=None):
  """
  Retrieve time_distributed layer and instatiate it as a model
  """ 
  for layer in model.layers:
    # if layer.name == distributedLayerName:
    if distributedLayerName in layer.name:
      distributedModel = layer
  var = distributedModel.variables
  conf = distributedModel.get_config()
  # instantiate patcher from config
  cnn = tf.keras.models.model_from_config(
      conf['layer'],
      custom_objects=custom_objects
  )
  # organising trained weights from var
  layerWeights = {}
  for layer in cnn.layers:
    layerVars = [v.name for v in layer.variables]
    layerWeights[layer.name] = [None for i in range(len(layerVars))]
    for idx in range(len(var)):
      varname = var[idx].name
      if varname in layerVars:
        # we must preserve vars layer order for the set_weight() to work
        varIdx = layerVars.index(varname)
        layerWeights[layer.name][varIdx] = var[idx]
  # set trained weights in instantiated cnn
  num_updated_layer = 0
  num_layer = 0
  for layer in cnn.layers:
    num_layer+=1
    if len(layerWeights[layer.name]) > 0:
      layer.set_weights(layerWeights[layer.name])
      num_updated_layer+=1
  print('Num layers and num updated : ',num_layer, num_updated_layer)
  return cnn

def gradCam(
    imTf,
    model,
    last_conv_layer_name = 'top_activation', #EffN2-S case
    classifier_layer_names =  ['avg_pool', 
                               'batch_normalization', 
                               'top_dropout',
                               'pred'], #EffN2-S case
    imSize = (128,128)
    ):
  preds = model.predict(imTf)
  img_array = imTf
  # Convolution model : input -> last conv layer's output (on which the class activation is computed)
  last_conv_layer  = model.get_layer(last_conv_layer_name)
  conv_model       = tf.keras.Model(model.inputs, last_conv_layer.output)
  # classifier_model : last conv layer's output / class activation -> final output : prediction
  classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name in classifier_layer_names:
      x = model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)

  # reccording conv model gradients
  with tf.GradientTape() as tape:
      last_conv_layer_output = conv_model(img_array)
      tape.watch(last_conv_layer_output)
      # Calcula la predicción con modelo clasificador, para la clase mas probable
      # Retrieve class prediction index and value
      preds = classifier_model(last_conv_layer_output)
      top_pred_index = tf.argmax(preds[0])
      print(top_pred_index)
      top_class_channel = preds[:, top_pred_index]
  # gradients of the last conv with respect to the input
  grads = tape.gradient(top_class_channel, last_conv_layer_output)
  # gradient average per filter layer
  pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)) # TODO : generalised to any number of channels
  # last conv output
  last_conv_layer_output = last_conv_layer_output[0]
  # gcam result
  heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

  heatmap = tf.squeeze(heatmap)
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
  vizGcam = resize(heatmap.numpy(), imSize)

  tf.keras.backend.clear_session()
  del grads
  del pooled_grads
  del tape
  tf.keras.backend.clear_session()
  return vizGcam



def guidedBackProp(
    preprocessed_img,
    model,
    include_relu    = False, # there are no relu in effnet
    include_swish   = True, # effnet case
    include_sigmoid = False,
    include_linear  = False,
    LAST_CONV_NAME = ''
    ):
   # Model until the last conv layer (where spatial info is preserved)
  gb_model = tf.keras.models.Model(
      inputs = [model.inputs],
      outputs = [model.get_layer(LAST_CONV_NAME).output]
  )

  # Activations for back prop gradient
  @tf.custom_gradient
  def guidedRelu(x):
    def grad(dy):
      return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad
  @tf.custom_gradient
  def guidedSwish(x):
    def grad(dy):
      return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.swish(x), grad
  @tf.custom_gradient
  def guidedLinear(x):
    def grad(dy):
      return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return x, grad
  @tf.custom_gradient
  def guidedSigmoid(x):
    def grad(dy):
      return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.sigmoid(x), grad

  # Overriding Actiivations ReLu with the guided gradient
  layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
  #numRelu = 0
  for layer in layer_dict:
    #print(layer.activation)
    if layer.activation == tf.keras.activations.relu and include_relu:
      layer.activation = guidedRelu
      #numRelu += 1
    elif layer.activation == tf.keras.activations.swish and include_swish:
      layer.activation = guidedSwish
    elif layer.activation == tf.keras.activations.sigmoid and include_sigmoid:
      layer.activation = guidedSigmoid
    elif layer.activation == tf.keras.activations.linear and include_linear:
      layer.activation = guidedLinear

  # Recording preprocessed input image during the forward path
  with tf.GradientTape() as tape:
    inputs = tf.cast(preprocessed_img, tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)[0]
  grads = tape.gradient(outputs,inputs)[0]

  # Guided back prop
  guided_back_prop =grads
  vizGb = np.dstack([guided_back_prop[:, :, cIdx] for cIdx in range(guided_back_prop.shape[2])])
  vizGb -= np.min(vizGb)
  vizGb /= vizGb.max()

  tf.keras.backend.clear_session()
  del grads
  #del guided_back_prop
  del tape
  tf.keras.backend.clear_session()
  return vizGb, guided_back_prop

def guidedGradCam(gradCamRes, guidedBackPropRes):
  gbGcam = np.dstack((
          guidedBackPropRes[:, :, 0] * gradCamRes,
          guidedBackPropRes[:, :, 1] * gradCamRes,
          guidedBackPropRes[:, :, 2] * gradCamRes,
      ))
  return gbGcam

def visual_explainer(
    model,
    preprocFunc = tf.keras.applications.efficientnet_v2.preprocess_input,
    last_conv_layer_name = 'top_activation', #EffN2-S case
    classifier_layer_names =  ['avg_pool', 
                               'batch_normalization', 
                               'top_dropout',
                               'pred'], #EffN2-S case
    include_relu    = False, # there are no relu in effnet
    include_swish   = True, # Effnet case
    include_sigmoid = False,
    include_linear  = False,
    imSize = (112,112),
    images = None
    ):

  imTf, imgArr, preprocessed_img = preprocIm(images, preprocFunc = preprocFunc)
  vizGcam = gradCam(imTf, model, last_conv_layer_name, classifier_layer_names, imSize)
  vizGb, guided_back_prop = guidedBackProp(preprocessed_img, model, include_relu, include_swish, include_sigmoid, include_linear, LAST_CONV_NAME = last_conv_layer_name)

  vizGbGcam = np.dstack((
          guided_back_prop[:, :, 0] * vizGcam,
          guided_back_prop[:, :, 1] * vizGcam,
          guided_back_prop[:, :, 2] * vizGcam,
      ))


  tf.keras.backend.clear_session()
  del guided_back_prop
  tf.keras.backend.clear_session()
  return imgArr, vizGcam, vizGb, vizGbGcam


def EnsembleExplainabilityRGB(ax              ,
                              rowIdx          ,
                              modName         ,
                              imgIdx          ,
                              alphaMask       ,
                              sigmaFact       ,
                              ds              ,
                              models          ,
                              maskBp          ,
                              bpCmam          , # twilight_shifted ; twilight_shifted
                              cmapMask        ,
                              fontsize,
                              label = '',
                              images = None    ,
                              imSize = (112,112)
                              ):
  maskTag = 'Gbp' if maskBp else 'Gd-GCAM'

  for idx in range(len(models)):
    mod = models[idx]
    label = np.argmax(label)
    pred = mod.predict(images)

    img, vizGcam, vizGb, vizGbGcam = visual_explainer(
        imgIdx                 = imgIdx,
        model                  = mod,
        images                 = images,
        dataset                = ds,
        preprocFunc            = tf.keras.applications.efficientnet_v2.preprocess_input,
        last_conv_layer_name   = 'top_activation', #EffN2-S case
        classifier_layer_names =  ['avg_pool', 'top_dropout','pred'], #EffN2-S case
        include_relu           = False, # there are no relu in effnet
        include_swish          = True,
        include_sigmoid        = False,
        include_linear         = False,
        imSize = imSize
        )

    # alpha    =alphaMask
    # sigmaFact = sigmaFact#0.5
    # mean = vizGbGcam.mean()
    # sigma = vizGbGcam.std()
    # min =  mean - sigma * sigmaFact #0.3
    # max =  mean + sigma * sigmaFact  #0.4
    # #vizGbGcam[(vizGbGcam>min) * (vizGbGcam<max)] = 0
    # vizGbGcam[(vizGbGcam<max)] = 0

    # finalVizGcam = dip.Overlay(magnetogram, mask,  color = [255,secCol,0])
    if idx == 0:
      vizGcamEns = vizGcam.astype('float32')
      vizGbEns = vizGb.astype('float32')
      vizGbGcamEns = vizGbGcam.astype('float32')
      predEns = pred
    else:
      vizGcamEns += vizGcam.astype('float32')
      vizGbEns += vizGb.astype('float32')
      vizGbGcamEns += vizGbGcam.astype('float32')
      predEns += pred

    tf.keras.backend.clear_session()
    del mod
    tf.keras.backend.clear_session()

  vizGcamEns = vizGcamEns / 10
  vizGbEns = vizGbEns / 10
  vizGbGcamEns = vizGbGcamEns / 10
  predEns = predEns/10

  ###################@@
  # # treshold = 0
  # #vizGbGcamEns = np.mean(vizGbGcamEns, axis = -1)
  # vizGbGcamEns = ( vizGbGcamEns - vizGbGcamEns.min()) / (vizGbGcamEns.max() - vizGbGcamEns.min()) #255*
  # # vizGbGcamEns[vizGbGcamEns<treshold] = 0
  # #vizGbEns  = np.mean(vizGbEns, axis = -1)
  # vizGbEns = ( vizGbEns - vizGbEns.min()) / (vizGbEns.max() - vizGbEns.min()) #255*
  # # vizGbEns[vizGbEns<treshold] = 0
  ###################@@
  vizGbGcamEns = np.mean(vizGbGcamEns, axis = -1)
  vizGbGcamEns = ( vizGbGcamEns - vizGbGcamEns.min()) / (vizGbGcamEns.max() - vizGbGcamEns.min()) #255*
  vizGbEns  = np.mean(vizGbEns, axis = -1)
  vizGbEns = ( vizGbEns - vizGbEns.min()) / (vizGbEns.max() - vizGbEns.min())

  ax[rowIdx,0].imshow(img)
  ax[rowIdx,1].imshow(img)#, cmap=plt.get_cmap('jet'))
  ax[rowIdx,1].imshow(vizGcamEns, alpha=alphaMask, cmap=plt.get_cmap('jet'))
  ax[rowIdx,2].imshow(vizGbEns, cmap=bpCmam)
  ax[rowIdx,3].imshow(vizGbGcamEns, cmap=bpCmam)

  # vizGbGcam[(vizGbGcamEns<max)] = 0
  # mask = np.zeros((128,128),dtype='bool')
  # mask[vizGbGcamEns>0] = 1
  if maskBp:
    mask = np.copy(vizGbEns)
  else:
    #vizGbGcam[(vizGbGcam<max)] = 0
    #mask = np.zeros((128,128),dtype='bool')
    #mask[vizGbGcam>0] = 1
    mask = np.copy(vizGbGcamEns)
  mean = mask.mean()
  sigma = mask.std()
  min =  mean - sigma * sigmaFact #0.3
  max =  mean + sigma * sigmaFact  #0.4
  mask[(mask<max)] = 0

  ax[rowIdx,4].imshow(img)#, cmap = plt.get_cmap('binary').reversed())
  ax[rowIdx,4].imshow(mask, alpha = alphaMask, cmap = cmapMask)


  ax[rowIdx,0].set_title(f'Model: {modName} \nPred : {predEns[0][0]:.2f}', fontsize=fontsize)
  ax[rowIdx,1].set_title(f'Grad-CAM', fontsize=fontsize)
  ax[rowIdx,2].set_title(f'GuidedBacProp', fontsize=fontsize)
  ax[rowIdx,3].set_title(f'Gd-GCAM', fontsize=fontsize)
  # ax[0,4].set_title('(1-2\u03C3)-GbCAM mask on Blos', fontsize=fontsize)
  ax[rowIdx,4].set_title(f'>(\u03BC+{sigmaFact}\u03C3)-{maskTag} mask on input', fontsize=fontsize)
  return predEns, vizGcamEns, vizGbEns, vizGbGcamEns
