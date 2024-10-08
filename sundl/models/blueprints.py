"""
Functions to instantiate tensorflow models
"""

import tensorflow as tf

from sundl.models.wrappers import reinstatiateOptim
from sundl.models.blocks import Cct_Block_Functional, CrossModalSpatialAttention



__all__ = ['build_pretrained_model',
           'build_pretrained_PatchCNN',
           'build_persistant_model'
           ]

def build_CNN_CrossModalAttention(
    num_classes = None,
    img_size = (6, 448, 448,1),
    alreadRgbStartIndices = [3], # starting indexes of channels to be processed as an rgb images
    tfModel = tf.keras.applications.efficientnet_v2.EfficientNetV2S, # rgb-cnn feature extractor
    hasAttention = True, # False for multimodal model without attention
    attentionDimProjType3D = True,
    residual_attention = True,
    num_attention_heads = 4, 
    attention_units = 64,
    pretainedWeight = True,
    loss = tf.keras.losses.MeanAbsoluteError(name='loss'), # tfloss or cme_mae , cme_rmse
    optimizer = 'adam',
    metrics= None,
    regression = True,
    scaledRegression = False,
    unfreeze_top_N = None,
    modelName = 'CmaCnn',
    ensure_residual_compat = True,
    feature_reduction = 32, # Not used if residual_attention is True and attentionType not None , if ensure_residual_compat is True
    replace_last_conv = False, # feature_reduction replace or add after it
    lastTfConv = 'top_conv',
    alpha = 0.5,
    globalPooling = True,
    scalarFeaturesSize = None,
    scalarAgregation = 'feature', # @['feature', 'baseline']
    compileModel = True,
    labelSize = None, # for regression oonly, for classification use num_classes
    preprocessing = None,
    l1_attention_reg = tf.keras.regularizers.l1(1e-3),
    **kwargs
):

  if scalarFeaturesSize is not None:
    image = tf.keras.layers.Input(shape=img_size, name='image')
    scalar_input =  tf.keras.layers.Input(shape=(scalarFeaturesSize), name='scalars')
  else:
    image = tf.keras.layers.Input(shape=img_size, name='image')
    
  if preprocessing is not None:
    preprocc_image = preprocessing(image)
  else:
    preprocc_image = image
    
  channels = []
  chanelIdx = 0

  while chanelIdx < img_size[0]:
    if chanelIdx not in alreadRgbStartIndices:
      channels.append(tf.expand_dims(tf.keras.layers.Conv2D(name = f'gray2RGB_{chanelIdx}',
                                                            filters = 3,
                                                            kernel_size = (3,3),
                                                            padding = 'same',
                                                            activation = None
                                                            )(preprocc_image[:,chanelIdx]),
                                      axis = 1)
      )
      chanelIdx += 1
    else:
      channels.append(tf.keras.layers.Permute((4, 2, 3, 1))(preprocc_image[:,chanelIdx:chanelIdx+3]))
      chanelIdx += 3
    print(channels[-1].shape)
    
  channels = tf.keras.layers.Concatenate(axis=1)(channels)
  
  print(channels.shape)
  
  
  extractor_input  = tf.keras.layers.Input(shape=(img_size[1],img_size[2],3), name='image')
  if pretainedWeight:
    feature_extractor = tfModel(include_top=False, input_tensor=extractor_input, weights="imagenet",**kwargs)
  else:
    feature_extractor = tfModel(include_top=False, input_tensor=extractor_input, weights=None,**kwargs)
    
  # Freeze the pretrained weights
  feature_extractor.trainable = not pretainedWeight
  if pretainedWeight and unfreeze_top_N is not None:
    # We unfreeze the top N layers while leaving BatchNorm layers frozen
    if unfreeze_top_N == 'all':
      for layer in feature_extractor.layers:
          layer.trainable = True
    else:
      ct = 0
      for layer in reversed(feature_extractor.layers):#[-unfreeze_top_N:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.trainable = True
          ct+=1
        if ct > unfreeze_top_N:
          break

  if residual_attention and hasAttention and ensure_residual_compat:
    feature_reduction  = num_attention_heads * attention_units
    print('WARNING  : "residual_attention" is "True", "feature_reduction" is ignored and set to "num_attention_heads * attention_units"')
  if feature_reduction is not None:
    for layer in feature_extractor.layers:
      if layer.name == lastTfConv:
        if replace_last_conv:
          preConvInput = layer.input
        else:
          preConvInput = layer.output
    # print(feature_reduction)
    if type(feature_reduction) == int:
      top_conv = tf.keras.layers.Conv2D(name = 'top_conv2',
                                  filters = feature_reduction,
                                  kernel_size = (3,3),
                                  padding='same',
                                  activation = None
                                  )(preConvInput)
    else:
      top_conv = feature_reduction(preConvInput)
    top_conv = tf.keras.layers.BatchNormalization(name =  f'top_bn2')(top_conv)
    top_conv = tf.keras.layers.Activation(activation='relu', name =  f'top_activation2')(top_conv)
    feature_extractor = tf.keras.models.Model(
        feature_extractor.input,
        top_conv
    )

  features = tf.keras.layers.TimeDistributed(feature_extractor)(channels)

  if hasAttention:
    # # Apply cross-modal-and-spatial self-attention
    # if attentionType == 'modal':
    #   attention_layer = CrossModalAttention(input_mode_dimension = features.shape[1], 
    #                                         spatial_dimension =  features.shape[2],
    #                                         num_attention_heads = num_attention_heads,
    #                                         attention_units = attention_units,
    #                                         residual = residual_attention,
    #                                         proj_type_3D = attentionDimProjType3D
    #                                         )
    #   # DIM : [batch, modes, attention_units * num_attention_heads]
    #   attention_output = attention_layer(features)
    #   # DIM : [batch, modes, 1, 1, attention_units * num_attention_heads]
    #   attention_output = tf.expand_dims(tf.expand_dims(attention_output, axis=2), axis=2)
    # else:
    #   print('WARNING : assumed attention type is modal_spatial')
    attention_layer = CrossModalSpatialAttention(
                                          input_mode_dimension = features.shape[1], 
                                          spatial_dimension =  features.shape[2],
                                          num_attention_heads = num_attention_heads,
                                          attention_units = attention_units,
                                          residual = residual_attention,
                                          proj_type_3D = attentionDimProjType3D,
                                          l1_reg = l1_attention_reg
                                          )
    # DIM : [batch, modes , height , width, num_attention_heads * attention_units]
    attention_output = attention_layer(features)
    x = attention_output
  else:
    x = features
    
  # Output layers
  if globalPooling:
    x = tf.keras.layers.AveragePooling3D((1, x.shape[2], x.shape[3]), name="avg_pool_vectorisation" )(x)
    x = tf.keras.layers.BatchNormalization()(x)
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


def build_persistant_model(
    loss="mae",
    metrics= None,
    encoder = None,
    regression = True, # unused but needed  for fn prototype compatibility (of builDS from ModelInstantier class)
    num_classes = 2,
    compileModel = True,
    prec = 'float32',
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
        output = tf.cast(output, dtype=prec)
      # output = tf.cast(output, dtype=prec)
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
    top_dropout_rate = 0.2,
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
    unfreeze_neurons = True,
    unfreeze_BN = True,
    modelName = 'PretrainedModel',
    feature_reduction = None,
    red_type = 'replace',
    lastTfConv = 'top_conv',
    alpha = 0.5,
    globalPooling = True,
    globalPoolingType = 'avg',
    scalarFeaturesSize = None,
    scalarAgregation = 'feature', # @['feature', 'baseline']
    compileModel = True,
    labelSize = None, # for regression oonly, for classification use num_classes
    preprocessing = None,
    pred_L1_reg=None,
    local_pred = False,
    top_dropout_rate = 0.2,
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
    model = tfModel(include_top=False, input_tensor=preprocc_image, weights="imagenet")#,**kwargs)
  else:
    model = tfModel(include_top=False, input_tensor=preprocc_image, weights=None)#,**kwargs)
    
  # Freeze the pretrained weights
  model.trainable = not pretainedWeight
  if pretainedWeight and unfreeze_top_N is not None:
    # We unfreeze the top N layers while leaving BatchNorm layers frozen
    if unfreeze_top_N == 'all':
      for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization) and unfreeze_BN:
          layer.trainable = True
        elif not isinstance(layer, tf.keras.layers.BatchNormalization) and unfreeze_neurons:
          layer.trainable = True
    else:
      ct = 0
      for layer in reversed(model.layers):#[-unfreeze_top_N:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization) and unfreeze_neurons:
          layer.trainable = True
          ct+=1
        elif isinstance(layer, tf.keras.layers.BatchNormalization) and unfreeze_BN:
          layer.trainable = True
          ct+=1
        if ct > unfreeze_top_N:
          break

  if feature_reduction is not None:
    for layer in model.layers:
      if layer.name == lastTfConv:
        if red_type == 'replace':
          preConvInput = layer.input
          tagLayerSfx = ''
        else:
          layer._name = 'pre-final-conv'
          preConvInput = layer.output
          # tagLayerSfx = '-f'
          tagLayerSfx = ''
    # print(feature_reduction)
    if type(feature_reduction) == int:
      if red_type == 'replace':
        preConvInput = tf.keras.layers.Activation(activation='swish', name =  f'pre_top_activation')(preConvInput)
      top_conv = tf.keras.layers.Conv2D(name = f'top_conv{tagLayerSfx}',
                                  filters = feature_reduction,
                                  kernel_size = (3,3),
                                  # strides = [1,1],
                                  # padding = [1,1],
                                  activation = None
                                  )(preConvInput)
    else:
      top_conv = feature_reduction(preConvInput)
    top_conv = tf.keras.layers.BatchNormalization(name =  f'top_bn{tagLayerSfx}')(top_conv)
    top_conv = tf.keras.layers.Activation(activation='swish', name =  f'top_activation{tagLayerSfx}')(top_conv)
    model = tf.keras.models.Model(
     model.input ,
     top_conv
    )
      
  x = model.output


  # if feature_reduction is not None:
  #   x = feature_reduction(x)

  # Output layers
  if local_pred:
    h = x.shape[1]
    w = x.shape[2]
    x = tf.keras.layers.Reshape((h*w, x.shape[-1]))(x)
    MLP = tf.keras.Sequential([
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation="sigmoid", name="local_pred", kernel_regularizer=pred_L1_reg)
    # Global average pooling
    ])
    x = tf.keras.layers.TimeDistributed(MLP)(x)
    x = tf.keras.layers.Reshape((h, w, x.shape[-1]))(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    output_final = tf.keras.layers.Flatten()(x)
    output = tf.concat([tf.cast(1,dtype=output_final.dtype)-output_final,output_final],axis=1)
  else:
    if globalPooling:
      if globalPoolingType == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool_vectorisation")(x)
      else:
        x = tf.keras.layers.GlobalMaxPooling2D(name="avg_pool_vectorisation")(x)
      x = tf.keras.layers.BatchNormalization()(x)
    else:
      x = tf.keras.layers.Flatten(name=f'flatten')(x)
    if scalarFeaturesSize is not None and scalarAgregation == 'feature':
      x = tf.keras.layers.Concatenate(axis=1)([x,tf.keras.layers.BatchNormalization()(scalar_input)])
    # x = tf.keras.layers.BatchNormalization()(x)
    # top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    if regression:
      if labelSize is None:
        labelSize = 1
      if scaledRegression:
        # Case where regression labels are scaled in [0,1] for potentially more stability)
        output = tf.keras.layers.Dense(labelSize, activation="sigmoid", name="pred", kernel_regularizer =  pred_L1_reg)(x)
      else:
        output = tf.keras.layers.Dense(labelSize, name="pred", kernel_regularizer =  pred_L1_reg )(x)
        # TODO : add baseeline case to otheer otpt kinds
        if scalarFeaturesSize is not None and scalarAgregation == 'baseline':
          output = tf.keras.layers.Add()([output, scalar_input])
    else:
      output = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred", kernel_regularizer =  pred_L1_reg)(x)
    
    
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

def build_LSTM(input_shape, # (#timestems, #feature)
               output_shape = None,
               lstm_units = [64],
               loss="mae",
               optimizer = 'adam',
               metrics= None,
               regression = True, # unused but needed  for fn prototype compatibility (of builDS from ModelInstantier class)
               num_classes = 2,
               compileModel = True,
               modelName = 'LSTM',
               preBatchNorm = True,
               dropoutLstm = 0.0,
               top_dropout_rate = None
               ):
  
  if output_shape is None:
    if regression:
      output_shape = 1
    else:
      output_shape = num_classes
  inputs = tf.keras.layers.Input(shape=input_shape, name='scalars')
  x = inputs
  if preBatchNorm:
    x = tf.keras.layers.BatchNormalization(name =  f'pre_bn')(x)
  for idx,nUnit in enumerate(lstm_units):
    if idx < len(lstm_units) - 1:
      x = tf.keras.layers.LSTM(nUnit, return_sequences=True, name = f'lstm_{idx}',dropout = dropoutLstm)(x)
    else:
      x = tf.keras.layers.LSTM(nUnit, name = f'lstm_{idx}', dropout = dropoutLstm)(x)

  if top_dropout_rate is not None: 
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
  if regression:
    output = tf.keras.layers.Dense(output_shape, name="pred")(x)
  else:
    output = tf.keras.layers.Dense(output_shape, activation="softmax", name="pred")(x)
    
  model = tf.keras.Model(inputs, output, name=modelName)
  if compileModel:
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
  return model
  

