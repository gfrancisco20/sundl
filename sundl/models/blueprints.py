"""
Functions to instantiate tensorflow models
"""

import tensorflow as tf

from sundl.models.wrappers import reinstatiateOptim

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
        output = tf.cast(output, dtype='uint8')
        output = tf.one_hot(output,num_classes)
      # output = tf.cast(output, dtype='float32')
      # output = output >=1
      # output = tf.cast(output, dtype=inputs.dtype)
      return output
  model = PersistantModel()
  if compileModel:
    model.compile(loss=loss, metrics=metrics)
  return model

def __build_pretrained_innerPatch(
    num_classes,
    img_size = (256, 256),
    tfModel = tf.keras.applications.efficientnet_v2.EfficientNetV2S,
    pretainedWeight = True,
    regression = False,
    scaledRegression = False,
    unfreeze_top_N = None,
    patche_output_type = 'pre_pred', # choose in ['pre_pred', 'flatten_features', 'feature_map']
    meth_patche_agg = 'avg',
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

  if patche_output_type == 'feature_map':
    patch_model = tf.keras.Model(input, model.output, name="Patch")
    #patch_output = model.output
  else:
    # Output layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
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
  return patch_model

def build_pretrained_PatchCNN(
    num_classes,
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
    **kwargs
):

  input = tf.keras.layers.Input(shape=(img_size[0], img_size[1], img_size[2]), name='input')

  #print('IN' , input.shape)
  patches = tf.image.extract_patches(input,
                                     sizes   = [1, patches_size[0], patches_size[1], 1],
                                     strides = [1, patches_size[0], patches_size[1], 1],
                                     rates   = [1, 1, 1, 1],
                                     padding='VALID'
                                     )
  if includeInterPatches:
    interPatches = tf.image.extract_patches(tf.keras.layers.Cropping2D(cropping=((0, patches_size[0]//2)))(input),
                                     sizes   = [1, patches_size[0], patches_size[1], 1],
                                     strides = [1, patches_size[0], patches_size[1], 1],
                                     rates   = [1, 1, 1, 1],
                                     padding='VALID'
                                     )
    print('interPatches', interPatches.shape)
  #print('EXT OUTPUT' , patches.shape)
  if shared_patcher == 'all':
    patches = tf.reshape(patches,shape=(-1,
                                        patches.shape[1]*patches.shape[2],
                                        patches_size[0],
                                        patches_size[1],
                                        patches_size[2]))
    if includeInterPatches:
      interPatches = tf.reshape(interPatches,shape=(-1,
                                        interPatches.shape[1]*interPatches.shape[2],
                                        patches_size[0],
                                        patches_size[1],
                                        patches_size[2]))
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
        patch = tf.reshape(patches[:,rIdx,cIdx,:],shape=(-1,patches_size[0],patches_size[1],patches_size[2]))
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
    **kwargs
):

  if scalarFeaturesSize is not None:
    image = tf.keras.layers.Input(shape=(img_size[0], img_size[1], img_size[2]), name='image')
    scalar_input =  tf.keras.layers.Input(shape=(scalarFeaturesSize), name='scalars')
  else:
    image = tf.keras.layers.Input(shape=(img_size[0], img_size[1], img_size[2]), name='image')

  if pretainedWeight:
    model = tfModel(include_top=False, input_tensor=image, weights="imagenet",**kwargs)
  else:
    model = tfModel(include_top=False, input_tensor=image, weights=None,**kwargs)
    
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