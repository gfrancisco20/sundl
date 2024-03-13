import tensorflow as tf
import numpy as np

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
                shape, 0, 1, seed=self.seed_generator
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
    image_size         = None,
    input_shape        = (224,224,1),
    num_heads          = 2 ,
    projection_dim     = 128,
    transformer_units  = [128, 128],
    transformer_layers = 2,
    tokenizer_config   = None,
    stochastic_depth_rate = 0.2,
    preprocessing = None,
    cross_channel_attention = False,
):
    
    inputs = tf.keras.layers.Input(input_shape)


    if image_size is None :
      image_size = input_shape[0]

    # Augment data.
    if preprocessing is None:
      preprocessing = tf.keras.Sequential(
      [
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        tf.keras.layers.RandomCrop(image_size, image_size),
        tf.keras.layers.RandomFlip("vertical"),
      ],
      name="preprocessing",
      )
    
    if len(input_shape)>3:
      channels = []
      
      # print(inputs[:,0].shape)
      for chanIdx in range(input_shape[0]):
        # print(chanIdx, input_shape[0])
        channels.append(preprocessing(inputs[:,chanIdx]))
      # print('OK')
      augmented = tf.stack(channels, axis=1)
      # print(augmented.shape)
    else:
      augmented = preprocessing(inputs)

    # Encode patches.
    if tokenizer_config is None:
      cct_tokenizer = CCTTokenizer()
    else:
      cct_tokenizer = CCTTokenizer(**tokenizer_config)
      
    if cross_channel_attention and len(input_shape) > 3:
      chanPatches = []
      for chanIdx in range(input_shape[0]):
        chanPatches.append(cct_tokenizer(augmented[:,chanIdx]))
      encoded_patches = tf.keras.layers.Concatenate(axis=1)(chanPatches)
    else:
      encoded_patches = cct_tokenizer(augmented)
    
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
#     image_size         = None,
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


#     if image_size is None :
#       image_size = input_shape[0]

#     # Augment data.
#     if preprocessing is None:
#       preprocessing = tf.keras.Sequential(
#       [
#         layers.Rescaling(scale=1.0 / 255),
#         layers.RandomCrop(image_size, image_size),
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
#       augmented = tf.stack(channels, axis=1)
#       # print(augmented.shape)
#     else:
#       augmented = preprocessing(inputs)

#     # Encode patches.
#     if tokenizer_config is None:
#       cct_tokenizer = CCTTokenizer()
#     else:
#       cct_tokenizer = CCTTokenizer(**tokenizer_config)
      
#     if cross_channel_attention and len(input_shape) > 3:
#       chanPatches = []
#       for chanIdx in range(input_shape[0]):
#         chanPatches.append(cct_tokenizer(augmented[:,chanIdx]))
#       encoded_patches = tf.keras.layers.Concatenate(axis=1)(chanPatches)
#     else:
#       encoded_patches = cct_tokenizer(augmented)
    
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
#     image_size         = None,
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
#     image_size         = image_size,
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