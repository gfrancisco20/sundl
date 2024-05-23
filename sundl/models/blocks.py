'''
Collection of usefull layers and functional/sequential blocks
'''

import tensorflow as tf
import numpy as np

import tensorflow as tf

class CrossModalAttention(tf.keras.layers.Layer):
  def __init__(self, 
                input_mode_dimension, # modal or also time dimension
                spatial_dimension,
                num_attention_heads, 
                attention_units, 
                residual = True, # WARNING : if True input feature dimension must be equua to num_attention_heads * attention_units
                proj_type_3D = True  # cross-modal projection
                ):
    super(CrossModalAttention, self).__init__()
    
    self.input_mode_dimension = input_mode_dimension
    self.spatial_dimension = spatial_dimension
    
    self.num_attention_heads = num_attention_heads
    
    self.attention_units = attention_units
    
    self.size = num_attention_heads * attention_units
    
    self.residual = residual
    
    self.proj_type_3D = proj_type_3D
    
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    
    # DIM : [batch, modes, num_features]
    self.spatial_reduction = tf.keras.Sequential([tf.keras.layers.AveragePooling3D((1,self.spatial_dimension,self.spatial_dimension)),
                                                  tf.keras.layers.Reshape((self.input_mode_dimension,-1))])
    
    # Define queries, keys, and values projection layers for input mode dimension
    # DIM : [batch, modes, num_attention_heads * attention_units]
    if self.proj_type_3D:
      self.input_mode_q = tf.keras.layers.Conv1D(filters=self.size, kernel_size=1)
      self.input_mode_k = tf.keras.layers.Conv1D(filters=self.size, kernel_size=1)
      self.input_mode_v = tf.keras.layers.Conv1D(filters=self.size, kernel_size=1)
    else:
      self.input_mode_q = tf.keras.layers.Dense(units=self.size)
      self.input_mode_k = tf.keras.layers.Dense(units=self.size)
      self.input_mode_v = tf.keras.layers.Dense(units=self.size)
    # Define output projection layer
    self.output_projection = tf.keras.layers.Dense(units=self.size) 
    
      
  def call(self, features):
    split_heads = tf.keras.Sequential([tf.keras.layers.Reshape(( -1, self.num_attention_heads, self.attention_units)),
                                       tf.keras.layers.Permute((2, 1, 3))
    ])
    
    # DIM : [batch, modes, num_features]
    features = self.spatial_reduction(features)
    
    # Project features to queries, keys, and values for input mode dimension
    # DIM : [batch, num_attention_heads, mode, attention_units]
    input_mode_q = split_heads(self.input_mode_q(features))
    input_mode_k = split_heads(self.input_mode_k(features))
    input_mode_v = split_heads(self.input_mode_v(features))

    # Compute self-attention for input mode dimension
    # DIM : [batch, num_attention_heads , modes, modes]
    input_mode_attention_weights = tf.matmul(input_mode_q, input_mode_k, transpose_b=True)
    input_mode_attention_weights /= tf.sqrt(tf.cast(self.attention_units),dtype=input_mode_q.dtype)
    input_mode_attention_weights = tf.nn.softmax(input_mode_attention_weights, axis=-1)
    
    # DIM : [batch, num_attention_heads, modes, attention_units]
    input_mode_attention_output = tf.matmul(input_mode_attention_weights, input_mode_v)
    # DIM : [batch, modes, attention_units * num_attention_heads]
    input_mode_attention_output =  tf.keras.layers.Permute([2, 1, 3])(input_mode_attention_output)
    input_mode_attention_output = tf.keras.layers.Reshape((-1, self.size))(input_mode_attention_output)

    
    # Project combined attention output
    projected_output = self.output_projection(input_mode_attention_output)
    projected_output = tf.keras.layers.Dropout(0.2)(projected_output)

    if self.residual:
      projected_output = self.layer_norm(features + projected_output)
    else:
      projected_output = self.layer_norm(projected_output)
  
    return projected_output


class CrossModalSpatialAttention(tf.keras.layers.Layer):
  def __init__(self, 
                input_mode_dimension, 
                spatial_dimension,
                num_attention_heads, 
                attention_units, 
                residual = True, # WARNING : if True input feature dimension must be equua to num_attention_heads * attention_units
                proj_type_3D = True # cross-modal projection
                ):
    super(CrossModalSpatialAttention, self).__init__()
    self.input_mode_dimension = input_mode_dimension
    self.spatial_dimension = spatial_dimension
    self.num_attention_heads = num_attention_heads
    self.attention_units = attention_units
    
    self.size = num_attention_heads * attention_units

    self.residual = residual
    self.proj_type_3D = proj_type_3D
    
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    # Define queries, keys, and values projection layers
    # DIM : [batch, modes * height * width , 3 * num_attention_heads * attention_units]
    if self.proj_type_3D:
      self.qkv_projection = tf.keras.Sequential([
          tf.keras.layers.Conv3D(filters=3 * self.size, kernel_size=1),
          tf.keras.layers.Reshape((self.input_mode_dimension * self.spatial_dimension * self.spatial_dimension, 3 * self.size))
      ])
    else:
      self.qkv_projection = tf.keras.Sequential([
          tf.keras.layers.Dense(3 * self.size),
          tf.keras.layers.Reshape((self.input_mode_dimension * self.spatial_dimension * self.spatial_dimension, 3 * self.size))
      ])
    
    # Define output projection layer
    self.output_projection = tf.keras.layers.Dense(self.size)
      
  def call(self, features):
    split_heads = tf.keras.Sequential([tf.keras.layers.Reshape(( -1, self.num_attention_heads, self.attention_units)),
                                       tf.keras.layers.Permute((2, 1, 3))
    ])
    merge_heads = tf.keras.Sequential([tf.keras.layers.Permute((2, 1, 3)),
                                      tf.keras.layers.Reshape(( -1, self.num_attention_heads * self.attention_units))
    ])                 
    
    # Compute queries, keys, and values
    # DIM : [batch, modes * height * width , 3 * num_attention_heads * attention_units]
    qkv = self.qkv_projection(features)
    
    # Split Q, K, and V for each attention head
    # DIM : [batch, modes * height * width , num_attention_heads * attention_units]  for each
    q, k, v = tf.split(qkv, 3, axis=-1)
    
    # DIM : [batch, num_attention_heads, modes * height * width, attention_units]
    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    
    # Compute self-attention for each attention head
    # DIM : [batch, num_attention_heads, modes * height * width, modes * height * width]
    attention_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.attention_units),dtype=q.dtype), axis=-1)
    
    # DIM : [batch, num_attention_heads, modes * height * width, attention_units]
    attention_output = tf.matmul(attention_weights, v)
    
    # Merge attention outputs from all attention heads
    # DIM : [batch, modes * height * width, num_attention_heads * attention_units]
    attention_output =  merge_heads(attention_output)
    
    # Project attention output
    # DIM : [batch, modes * height * width, num_attention_heads * attention_units]
    output = self.output_projection(attention_output)
    
    # Original Shape
    # DIM : [batch, modes , height , width, num_attention_heads * attention_units]
    output = tf.keras.layers.Reshape((self.input_mode_dimension, self.spatial_dimension, self.spatial_dimension, self.size))(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    
    if self.residual:
      output = self.layer_norm(features + output)
    else:
      output = self.layer_norm(output)
      
    return output

class CCTTokenizer(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_output_channels=[64, 128],
        positional_emb= True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        num_conv_layers = len(num_output_channels)

        # This is our tokenizer.
        self.conv_model = tf.keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                tf.keras.layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(tf.keras.layers.ZeroPadding2D(padding))
            self.conv_model.add(
                tf.keras.layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (
                -1,
                tf.shape(outputs)[1] * tf.shape(outputs)[2],
                tf.shape(outputs)[-1],
            ),
        )
        return reshaped


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = tf.keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = tf.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape

class SequencePooling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = tf.keras.layers.Dense(1)

    def call(self, x):
        attention_weights = tf.math.softmax(self.attention(x), axis=1)
        attention_weights = tf.transpose(attention_weights, perm=(0, 2, 1))
        weighted_representation =  tf.linalg.matmul(attention_weights, x)
        return tf.squeeze(weighted_representation, -2)

# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = tf.random.Generator.from_seed(49) # keras.random.SeedGenerator(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape, 0, 1, #seed=self.seed_generator
            )
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.keras.activations.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x
  

      
def Cct_Block_Functional(
    img_size         = None,
    input_shape        = (224,224,1),
    num_heads          = 2 ,
    projection_dim     = 128,
    transformer_units  = [128, 128],
    transformer_layers = 2,
    tokenizer_config   = None,
    stochastic_depth_rate = 0.2,
    preprocessing = None,
    preprocessChannelsIndependantly = False,
    cross_channel_attention = False,
):
    
    inputs = tf.keras.layers.Input(input_shape)


    if img_size is None :
      img_size = input_shape[0]

    # Augment data.
    if preprocessing is None:
      preprocessing = tf.keras.Sequential(
      [
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        tf.keras.layers.RandomCrop(img_size, img_size),
        tf.keras.layers.RandomFlip("vertical"),
      ],
      name="preprocessing",
      )
    
    if len(input_shape)>3:
      if preprocessChannelsIndependantly:
        channels = []
        
        # print(inputs[:,0].shape)
        for chanIdx in range(input_shape[0]):
          # print(chanIdx, input_shape[0])
          channels.append(preprocessing(inputs[:,chanIdx]))
        # print('OK')
        preproc_input = tf.stack(channels, axis=1)
        # print(preproc_input.shape)
      else:
        # inputs = tf.transpose(inputs, [0, 4, 2, 3, 1])
        # permutIdxs = [0, 4, 2, 3, 1]
        permutIdxs = [4, 2, 3, 1]
        preproc_input = tf.keras.layers.Permute(permutIdxs)(inputs)
        preproc_input = preproc_input[:,0]
        preproc_input = preprocessing(preproc_input)
        # preproc_input = tf.expand_dims(preproc_input, axis=1)
        preproc_input = tf.keras.layers.Reshape([1]+list(preproc_input.shape[1:]))(preproc_input)
        # preproc_input = tf.transpose(preproc_input, [0, 4, 2, 3, 1])
        preproc_input = tf.keras.layers.Permute(permutIdxs)(preproc_input)
    else:
      preproc_input = preprocessing(inputs)

    # Encode patches.
    if tokenizer_config is None:
      cct_tokenizer = CCTTokenizer()
    else:
      cct_tokenizer = CCTTokenizer(**tokenizer_config)
      
    if cross_channel_attention and len(input_shape) > 3:
      chanPatches = []
      for chanIdx in range(input_shape[0]):
        chanPatches.append(cct_tokenizer(preproc_input[:,chanIdx]))
      encoded_patches = tf.keras.layers.Concatenate(axis=1)(chanPatches)
    else:
      encoded_patches = cct_tokenizer(preproc_input)
    
    # Apply positional embedding.
    if cct_tokenizer.positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    weighted_representation = SequencePooling()(representation)

    # self.weighted_representation = weighted_representation
    
    return tf.keras.Model(inputs, weighted_representation, name="cct_block")

# class Cct_Block(layers.Layer):
#   def __init__(self, 
#     img_size         = None,
#     input_shape        = (224,224,1),
#     num_heads          = 2 ,
#     projection_dim     = 128,
#     transformer_units  = [128, 128],
#     transformer_layers = 2,
#     tokenizer_config   = None,
#     stochastic_depth_rate = 0.2,
#     preprocessing = None,
#     cross_channel_attention = False,
#     **kwargs
# ):
#     super().__init__(**kwargs)
    
#     inputs = tf.keras.layers.Input(input_shape)


#     if img_size is None :
#       img_size = input_shape[0]

#     # Augment data.
#     if preprocessing is None:
#       preprocessing = tf.keras.Sequential(
#       [
#         layers.Rescaling(scale=1.0 / 255),
#         layers.RandomCrop(img_size, img_size),
#         layers.RandomFlip("vertical"),
#       ],
#       name="preprocessing",
#       )
    
#     if len(input_shape)>3:
#       channels = []
      
#       # print(inputs[:,0].shape)
#       for chanIdx in range(input_shape[0]):
#         # print(chanIdx, input_shape[0])
#         channels.append(preprocessing(inputs[:,chanIdx]))
#       # print('OK')
#       preproc_input = tf.stack(channels, axis=1)
#       # print(preproc_input.shape)
#     else:
#       preproc_input = preprocessing(inputs)

#     # Encode patches.
#     if tokenizer_config is None:
#       cct_tokenizer = CCTTokenizer()
#     else:
#       cct_tokenizer = CCTTokenizer(**tokenizer_config)
      
#     if cross_channel_attention and len(input_shape) > 3:
#       chanPatches = []
#       for chanIdx in range(input_shape[0]):
#         chanPatches.append(cct_tokenizer(preproc_input[:,chanIdx]))
#       encoded_patches = tf.keras.layers.Concatenate(axis=1)(chanPatches)
#     else:
#       encoded_patches = cct_tokenizer(preproc_input)
    
#     # Apply positional embedding.
#     if cct_tokenizer.positional_emb:
#         sequence_length = encoded_patches.shape[1]
#         encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
#             encoded_patches
#         )

#     # Calculate Stochastic Depth probabilities.
#     dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

#     # Create multiple layers of the Transformer block.
#     for i in range(transformer_layers):
#         # Layer normalization 1.
#         x1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

#         # Create a multi-head attention layer.
#         attention_output = tf.keras.layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=projection_dim, dropout=0.1
#         )(x1, x1)

#         # Skip connection 1.
#         attention_output = StochasticDepth(dpr[i])(attention_output)
#         x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

#         # Layer normalization 2.
#         x3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x2)

#         # MLP.
#         x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

#         # Skip connection 2.
#         x3 = StochasticDepth(dpr[i])(x3)
#         encoded_patches = tf.keras.layers.Add()([x3, x2])

#     # Apply sequence pooling.
#     representation = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
#     weighted_representation = SequencePooling()(representation)

#     # self.weighted_representation = weighted_representation
    
#     self.cctBlock = tf.keras.Model(inputs, weighted_representation, name="cct_block")

#   def call(self, images):
#     return self.cctBlock(images)
  
# def build_cct_multiModal(
#     img_size         = None,
#     input_shape        = (224,224,1),
#     num_heads          = 2 ,
#     projection_dim     = 128,
#     transformer_units  = [128, 128],
#     transformer_layers = 2,
#     tokenizer_config   = None,
#     stochastic_depth_rate = 0.2,
#     preprocessing = None,
#     cross_channel_attention = False,
#     cross_channel_cct = False,
#     regression = True,
#     num_classes = 2
# ):
  
#   if cross_channel_attention and cross_channel_cct:
#     raise Exception(f'Cannot use both `cross_channel_attention` and `cross_channel_cct`')
  
#   if cross_channel_cct and len(input_shape) > 3:
#     cct_input_shape = input_shape[1:]
#   else:
#     cct_input_shape = input_shape
    
#   # print(cct_input_shape)
  
#   Cct = Cct_Block_Functional(
#     img_size         = img_size,
#     input_shape        = cct_input_shape,
#     num_heads          = num_heads ,
#     projection_dim     = projection_dim,
#     transformer_units  = transformer_units,
#     transformer_layers = transformer_layers,
#     tokenizer_config   = tokenizer_config,
#     stochastic_depth_rate = stochastic_depth_rate,
#     preprocessing = preprocessing,
#     cross_channel_attention = cross_channel_attention)
  
#   input = tf.keras.layers.Input(input_shape)
  
#   if cross_channel_cct:
#     print('STaRRT')
#     x = tf.keras.layers.TimeDistributed(Cct)(input)
#     x  = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])
#   else:
#     x = Cct(input)
    
#   if regression:
#     output = tf.keras.layers.Dense(1, name="pred")(x)
#   else:
#     output = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
  
#   model = tf.keras.Model(input, output, name="Patch")
  
#   return model