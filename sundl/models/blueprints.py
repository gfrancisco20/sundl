"""
Functions to instantiate tensorflow models
"""

import tensorflow as tf

from sundl.models.wrappers import reinstatiateOptim
from sundl.models.blocks import Cct_Block_Functional



__all__ = ['build_pretrained_model',
           'build_pretrained_PatchCNN',
           'build_persistant_model'
           ]

def build_persistant_model(
    loss="mae",
    metrics= None,
    encoder = None,
    regression = True, # unused but needed  for fn prototype compatibility (of builDS from ModelInstantier class)
    num_classes = 2,
    compileModel = True,
    **kwargs
):
  class PersistantModel(tf.keras.Model):
    def __init__(self):
      super().__init__()
      self.encoder = encoder
      self.regression = regression

    def call(self, inputs):
      # inputs = tf.cast(tf.expand_dims(inputs, axis=-1),'int32')
      # inputs = tf.expand_dims(inputs, axis=-1)
      # output =  tf.keras.layers.Concatenate(axis=1)([1-inputs,inputs])
      # if self.encoder is not None:
      #   inputs = self.encoder(inputs)
      output = inputs
      if not regression:
        if output.shape[1] != num_classes:
          output = tf.cast(output, dtype='uint8')
          output = tf.one_hot(output,num_classes)
        output = tf.cast(output, dtype='float32')
      # output = tf.cast(output, dtype='float32')
      # output = output >=1
      # output = tf.cast(output, dtype=inputs.dtype)
      return output
  model = PersistantModel()
  if compileModel:
    model.compile(loss=loss, metrics=metrics)
  return model

def __build_pretrained_innerPatch(
    num_classes = None,
    img_size = (256, 256),
    tfModel = tf.keras.applications.efficientnet_v2.EfficientNetV2S,
    pretainedWeight = True,
    regression = False,
    scaledRegression = False,
    unfreeze_top_N = None,
    patche_output_type = 'pre_pred', # choose in ['pre_pred', 'flatten_features', 'feature_map']
    meth_patche_agg = 'avg',
    feature_reduction = None,
    lastTfConv = 'top_conv',
    **kwargs
):
  input = tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3))
  if pretainedWeight:
    model = tfModel(include_top=False, input_tensor=input, weights="imagenet",**kwargs)
  else:
    model = tfModel(include_top=False, input_tensor=input, weights=None,**kwargs)
  # Freeze the pretrained weights
  model.trainable = not pretainedWeight
  if pretainedWeight and unfreeze_top_N is not None:
    # We unfreeze the top N layers while leaving BatchNorm layers frozen
    if unfreeze_top_N == 'all':
      for layer in model.layers:
          if not isinstance(layer, tf.keras.layers.BatchNormalization):
              layer.trainable = True
    else:
      for layer in model.layers[-unfreeze_top_N:]:
          if not isinstance(layer, tf.keras.layers.BatchNormalization):
              layer.trainable = True

  if feature_reduction is not None:
    for layer in model.layers:
      if layer.name == lastTfConv:
        preConvInput = layer.input
    # print(feature_reduction)
    if type(feature_reduction) == int:
      top_conv = tf.keras.layers.Conv2D(name = 'top_conv',
                                  filters = feature_reduction,
                                  kernel_size = (3,3),
                                  # strides = [1,1],
                                  # padding = [1,1],
                                  activation = None
                                  )(preConvInput)
    else:
      top_conv = feature_reduction(preConvInput)
    top_conv = tf.keras.layers.BatchNormalization(name =  f'top_bn')(top_conv)
    top_conv = tf.keras.layers.Activation(activation='relu', name =  f'top_activation')(top_conv)
    model = tf.keras.models.Model(
     model.input ,
     top_conv
    )
  x = model.output

  if patche_output_type == 'feature_map':
    patch_model = tf.keras.Model(input, x, name="Patch")
    #patch_output = model.output
  else:
    # Output layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    # flatten_features case output :
    output = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    if patche_output_type == 'pre_pred':
      if regression:
        if scaledRegression:
          # Case where regression labels are scaled in [0,1] for potentially more stability)
          output = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(output)
        else:
          output = tf.keras.layers.Dense(1, name="pred")(output)
      else:
        if meth_patche_agg in ['c1d', 'c1d_relu', 'c1d_lin']:
          output = tf.keras.layers.Dense(num_classes-1, activation="relu", name="pred")(output)
        elif meth_patche_agg in ['avg', 'max']:
          output = tf.keras.layers.Dense(num_classes-1, activation="sigmoid", name="pred")(output)
    patch_model = tf.keras.Model(input, output, name="Patch")
    #patch_output = output
    
    def compute_output_shape(self, input_shape): 
      return self.output.shape
    import types
    patch_model.compute_output_shape = types.MethodType(compute_output_shape, patch_model)
        
  return patch_model

def build_pretrained_PatchCNN(
    num_classes = None,
    img_size = (512, 1024, 3),
    patches_size = (256, 256, 3),
    tfModel = tf.keras.applications.efficientnet_v2.EfficientNetV2S,
    pretainedWeight = True,
    loss="binary_crossentropy",
    optimizer = 'adam',
    metrics= None,
    regression = False,
    scaledRegression = False,
    unfreeze_top_N = None,
    patche_output_type = 'pre_pred', # choose in ['pre_pred', 'flatten_features', 'feature_map']
    shared_patcher = 'no', # choose in ['no', 'all', 'limb_vs_center']
    limb_patches = [(0,0),(0,3),(1,0),(1,3)] , # def val for 256 patches
    meth_patche_agg = 'avg', # choose in ['c1d_lin', 'c1d_relu', 'avg', 'c3d', 'max', 'sum'] * c3d only if patche_output_type=feature_map
    includeInterPatches = False, # include patches covering the vertical intersections of basic patches
    compileModel = True,
    feature_reduction = None,
    lastTfConv = 'top_conv',
    preprocessing = None,
    **kwargs
):

  input = tf.keras.layers.Input(shape=(img_size[0], img_size[1], img_size[2]), name='input')
  
  if preprocessing is not None:
    x = preprocessing(input)
  else:
    x = input

  #print('IN' , input.shape)
  # backward compatibility
  # class Extract_patches_Layer(tf.keras.layers.Layer):
  #   def __init__(self):
  #     super().__init__()
  #   def call(self, x):
  #       return tf.image.extract_patches(x,
  #                                    sizes   = [1, patches_size[0], patches_size[1], 1],
  #                                    strides = [1, patches_size[0], patches_size[1], 1],
  #                                    rates   = [1, 1, 1, 1],
  #                                    padding='VALID'
  #                                    )
  #   def compute_output_shape(self, input_shape):
  #       nRow = input_shape[1] // patches_size[0]
  #       nCol = input_shape[2] // patches_size[1]
  #       return (input_shape[0], nRow, nCol, patches_size[0]*patches_size[1]*input_shape[3])
  # patches = Extract_patches_Layer()(x)
  patches = tf.image.extract_patches(x,
                                     sizes   = [1, patches_size[0], patches_size[1], 1],
                                     strides = [1, patches_size[0], patches_size[1], 1],
                                     rates   = [1, 1, 1, 1],
                                     padding='VALID'
                                     )
  if includeInterPatches:
    # interPatches =  Extract_patches_Layer()(tf.keras.layers.Cropping2D(cropping=((0, patches_size[0]//2)))(x))
    interPatches = tf.image.extract_patches(tf.keras.layers.Cropping2D(cropping=((0, patches_size[0]//2)))(x),
                                     sizes   = [1, patches_size[0], patches_size[1], 1],
                                     strides = [1, patches_size[0], patches_size[1], 1],
                                     rates   = [1, 1, 1, 1],
                                     padding='VALID'
                                     )
    print('interPatches', interPatches.shape)
  #print('EXT OUTPUT' , patches.shape)
  if shared_patcher == 'all':
    patches = tf.keras.layers.Reshape(target_shape = ( #tf.reshape(patches,shape=(-1,
                                        patches.shape[1]*patches.shape[2],
                                        patches_size[0],
                                        patches_size[1],
                                        patches_size[2])) (patches)
    if includeInterPatches:
      interPatches = tf.keras.layers.Reshape(target_shape = ( #tf.reshape(interPatches,shape=(-1,
                                        interPatches.shape[1]*interPatches.shape[2],
                                        patches_size[0],
                                        patches_size[1],
                                        patches_size[2])) (interPatches)
      print('reshaped interPatches', interPatches.shape)
      patches = tf.keras.layers.Concatenate(axis=1)([patches,interPatches])
      print('final patches', patches.shape)
    #print('RS ts OUUTPUT' , patches.shape)
    patch_model = __build_pretrained_innerPatch(
          num_classes,
          patches_size,
          tfModel,
          pretainedWeight,
          regression,
          scaledRegression,
          unfreeze_top_N,
          patche_output_type,
          meth_patche_agg = meth_patche_agg,
          feature_reduction = feature_reduction,
          lastTfConv = lastTfConv,
          **kwargs)
    patches_outputs =  tf.keras.layers.TimeDistributed(patch_model)(patches)
    #print('TS OUUTPUT' , patches_outputs.shape)
    #TS OUUTPUT (None, 8, 2)
    #PTS OUUTPUT (None, 2)
    patches_outputs = [patches_outputs[:,ptIdx,:] for ptIdx in range(patches_outputs.shape[1])]
  else:
    patches_outputs = []
    #patches_idxs = []
    for rIdx, row in enumerate(range(patches.shape[1])):
      for cIdx, col in enumerate(range(patches.shape[2])):
        #patches_idxs.append((rIdx,cIdx))
        # patch = tf.reshape(patches[:,rIdx,cIdx,:],shape=(-1,patches_size[0],patches_size[1],patches_size[2]))
        patch =  tf.keras.layers.Reshape(target_shape = (patches_size[0],patches_size[1],patches_size[2]))(patches[:,rIdx,cIdx,:])
        #print('RS OUUTPUT' , patch.shape)
        patch_model = __build_pretrained_innerPatch(
          num_classes,
          patches_size,
          tfModel,
          pretainedWeight,
          regression,
          scaledRegression,
          unfreeze_top_N,
          patche_output_type,
          meth_patche_agg = meth_patche_agg,
          feature_reduction = feature_reduction,
          lastTfConv = lastTfConv,
          **kwargs)
        # renaming layers to get unique names
        patch_model._name = patch_model._name + f'_p{rIdx}x{cIdx}'
        for i, layer in enumerate(patch_model.layers):
          layer._name = layer._name + f'_p{rIdx}x{cIdx}'
        patches_outputs.append(patch_model(patch))
        #print('PTS OUUTPUT' , patches_outputs[0].shape)

  if meth_patche_agg == 'avg':
    output = tf.keras.layers.Average()(patches_outputs)
  elif meth_patche_agg == 'sum':
    output = tf.keras.layers.Add()(patches_outputs)
  elif meth_patche_agg == 'max':
    # output = tf.keras.layers.Max()(patches_outputs)
    output = tf.keras.layers.maximum(patches_outputs)
  if not regression:
    if len(output.shape) == 1:
      output = tf.expand_dims(output, -1)
    output = tf.keras.layers.Concatenate(axis=-1)([1-output,output])
  elif meth_patche_agg in ['c1d','c1d_relu']:
    patches_outputs = [tf.expand_dims(tf.expand_dims(p, -1),-1) for p in patches_outputs]
    x = tf.keras.layers.Concatenate(axis=2)(patches_outputs)
    print(x.shape)
    output = tf.keras.layers.Conv1D(filters     = 1,
                                kernel_size = 8,
                                activation  = 'relu',
                                input_shape = x.shape[1:]
                                )(x)[:,:,0,0]

    output = tf.keras.layers.Activation('softmax')(output)
  elif meth_patche_agg == 'c1d_lin':
    patches_outputs = [tf.expand_dims(tf.expand_dims(p, -1),-1) for p in patches_outputs]
    x = tf.keras.layers.Concatenate(axis=2)(patches_outputs)
    print(x.shape)
    output = tf.keras.layers.Conv1D(filters     = 1,
                                kernel_size = 8,
                                activation  = 'linear',
                                input_shape = x.shape[1:]
                                )(x)[:,:,0,0]
    output = tf.keras.layers.Activation('softmax')(output)
  model = tf.keras.Model(input, output, name="PremadePatchConv")
  # TEMPORARY FIX TO ERROR BELOW ##################################
  # todo : add other optim handler if needed + add other optim params if needed
  # error : KeyError: 'The optimizer cannot recognize variable...
  # haapens in loops of CV
  if compileModel:
    optimizer = reinstatiateOptim(optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model




def build_pretrained_model(
    num_classes = None,
    img_size = (448, 448, 3),
    tfModel = tf.keras.applications.efficientnet_v2.EfficientNetV2S,
    pretainedWeight = True,
    loss=tf.keras.losses.MeanAbsoluteError(name='loss'), # tfloss or cme_mae , cme_rmse
    optimizer = 'adam',
    metrics= None,
    regression = True,
    scaledRegression = False,
    unfreeze_top_N = None,
    modelName = 'PretrainedModel',
    feature_reduction = None,
    lastTfConv = 'top_conv',
    alpha = 0.5,
    globalPooling = True,
    scalarFeaturesSize = None,
    scalarAgregation = 'feature', # @['feature', 'baseline']
    compileModel = True,
    labelSize = None, # for regression oonly, for classification use num_classes
    preprocessing = None,
    **kwargs
):

  if scalarFeaturesSize is not None:
    image = tf.keras.layers.Input(shape=(img_size[0], img_size[1], img_size[2]), name='image')
    scalar_input =  tf.keras.layers.Input(shape=(scalarFeaturesSize), name='scalars')
  else:
    image = tf.keras.layers.Input(shape=(img_size[0], img_size[1], img_size[2]), name='image')
    
  if preprocessing is not None:
    preprocc_image = preprocessing(image)
  else:
    preprocc_image = image

  if pretainedWeight:
    model = tfModel(include_top=False, input_tensor=preprocc_image, weights="imagenet",**kwargs)
  else:
    model = tfModel(include_top=False, input_tensor=preprocc_image, weights=None,**kwargs)
    
  # Freeze the pretrained weights
  model.trainable = not pretainedWeight
  if pretainedWeight and unfreeze_top_N is not None:
    # We unfreeze the top N layers while leaving BatchNorm layers frozen
    if unfreeze_top_N == 'all':
      for layer in model.layers:
          layer.trainable = True
    else:
      ct = 0
      for layer in reversed(model.layers):#[-unfreeze_top_N:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.trainable = True
          ct+=1
        if ct > unfreeze_top_N:
          break

  if feature_reduction is not None:
    for layer in model.layers:
      if layer.name == lastTfConv:
        preConvInput = layer.input
    # print(feature_reduction)
    if type(feature_reduction) == int:
      top_conv = tf.keras.layers.Conv2D(name = 'top_conv',
                                  filters = feature_reduction,
                                  kernel_size = (3,3),
                                  # strides = [1,1],
                                  # padding = [1,1],
                                  activation = None
                                  )(preConvInput)
    else:
      top_conv = feature_reduction(preConvInput)
    top_conv = tf.keras.layers.BatchNormalization(name =  f'top_bn')(top_conv)
    top_conv = tf.keras.layers.Activation(activation='relu', name =  f'top_activation')(top_conv)
    model = tf.keras.models.Model(
     model.input ,
     top_conv
    )
    x = model.output


  # if feature_reduction is not None:
  #   x = feature_reduction(x)

  # Output layers
  if globalPooling:
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool_vectorisation")(x)
    x = tf.keras.layers.BatchNormalization()(x)
  else:
    x = tf.keras.layers.Flatten(name=f'flatten')(x)
  if scalarFeaturesSize is not None and scalarAgregation == 'feature':
    x = tf.keras.layers.Concatenate(axis=1)([x,tf.keras.layers.BatchNormalization()(scalar_input)])
  # x = tf.keras.layers.BatchNormalization()(x)
  top_dropout_rate = 0.2
  x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
  if regression:
    if labelSize is None:
      labelSize = 1
    if scaledRegression:
      # Case where regression labels are scaled in [0,1] for potentially more stability)
      output = tf.keras.layers.Dense(labelSize, activation="sigmoid", name="pred")(x)
    else:
      output = tf.keras.layers.Dense(labelSize, name="pred", )(x)
      # TODO : add baseeline case to otheer otpt kinds
      if scalarFeaturesSize is not None and scalarAgregation == 'baseline':
        output = tf.keras.layers.Add()([output, scalar_input])
  else:
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
  
    
  # Compile
  if scalarFeaturesSize is not None:
    model = tf.keras.Model((image,scalar_input), output, name=modelName)
  else:
    model = tf.keras.Model(image, output, name=modelName)
  if compileModel:
    optimizer = reinstatiateOptim(optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model
  
def build_cct_multiModal(
    img_size         = None,
    input_shape        = (224,224,1), # if multimodal add channel dimension at the begining, e.g. (5, 224,224,1)
    num_heads          = 2 ,
    projection_dim     = 128,
    transformer_units  = [128, 128],
    transformer_layers = 2,
    tokenizer_config   = None,
    stochastic_depth_rate = 0.2,
    preprocessing = None,
    cross_channel_attention = False, # merge multimodal features before transformer layers
    cross_channel_cct       = False, # merge multimodal features at the end of the CCT (before final MLP)
    regression = True,
    num_classes = 2,
    optimizer = None,
    loss = None,
    metrics = None,
    compileModel = False,
    scalarFeaturesSize = None,
    labelSize = 1,
    modelName = 'Cct',
    preprocessChannelsIndependantly = False,
    **kwargs
):
  
  if cross_channel_attention and cross_channel_cct:
    raise Exception(f'Cannot use both `cross_channel_attention` and `cross_channel_cct`')
  
  if cross_channel_cct and len(input_shape) > 3:
    cct_input_shape = input_shape[1:]
  else:
    cct_input_shape = input_shape
    
  print('input_shape', input_shape)
  print('cct_input_shape', cct_input_shape)
  
  Cct = Cct_Block_Functional(
    img_size         = img_size,
    input_shape        = cct_input_shape,
    num_heads          = num_heads ,
    projection_dim     = projection_dim,
    transformer_units  = transformer_units,
    transformer_layers = transformer_layers,
    tokenizer_config   = tokenizer_config,
    stochastic_depth_rate = stochastic_depth_rate,
    preprocessing = preprocessing,
    preprocessChannelsIndependantly = preprocessChannelsIndependantly,
    cross_channel_attention = cross_channel_attention)
  
  if scalarFeaturesSize is not None:
    image = tf.keras.layers.Input(shape=input_shape, name='image')
    scalar_input =  tf.keras.layers.Input(shape=(scalarFeaturesSize), name='scalars')
  else:
    image = tf.keras.layers.Input(input_shape)
  
  if cross_channel_cct:
    x = tf.keras.layers.TimeDistributed(Cct)(image)
    x  = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])
  else:
    x = Cct(image)
    
  if scalarFeaturesSize is not None:
    x = tf.keras.layers.Concatenate(axis=1)([x,tf.keras.layers.BatchNormalization()(scalar_input)])
    
  if regression:
    output = tf.keras.layers.Dense(labelSize, name="pred")(x)
  else:
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
  
  if scalarFeaturesSize is not None:
    model = tf.keras.Model((image,scalar_input), output, name=modelName)
  else:
    model = tf.keras.Model(image, output, name=modelName)
  
  if compileModel:
    optimizer = reinstatiateOptim(optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  
  return model

