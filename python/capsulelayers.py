# -*- coding: utf-8 -*-
"""
@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland
"""

import tensorflow as tf   
from tensorflow.keras import layers, initializers
from tensorflow.keras import backend as K


"""
    File 8 
    
    This file is used to ......
    
"""

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=tf.shape(x)[1])

        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param num_routing: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=(self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule),
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
          # Expand the input in axis=1, tile in that axis to num_capsule, and 
          # expands another axis at the end to prepare the multiplication with W.
          inputs_expand = tf.expand_dims(inputs, 1)
          inputs_tiled  = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
          inputs_tiled  = tf.expand_dims(inputs_tiled, 4)
        
          # Compute `W * inputs` by scanning inputs_tiled on dimension 0 (map_fn).
          inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)     
        
          # Routing algorithm
          # The prior for coupling coefficient, initialized as zeros.
          b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, 
                              self.input_num_capsule, 1, 1])
        
          assert self.num_routing > 0, 'The routings should be > 0.'
          for i in range(self.num_routing):
            # Apply softmax to the axis with `num_capsule`
            c = layers.Softmax(axis=1)(b)
        
            # Compute the weighted sum of all the predicted output vectors.
            # The function `multiply` will broadcast axis=3 in c to dim_capsule.
            # Then sum along the input_num_capsule
            # Then apply squash along the dim_capsule
            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)  # [None, 10, 1, 16, 1]
        
            if i < self.num_routing - 1:
              # Update the prior b.
              # Multiply the outputs with the weighted_inputs (inputs_hat) and add  
              # it to the prior b.  
              outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
              agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
              b = tf.add(b, agreement)
        
          # End Routing algorithm 
          
          # Squeeze the outputs to remove useless axis:
          outputs = tf.squeeze(outputs, [2, 4])
          
          return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv1D `n_channels` times and concatenate all capsules

    """
    output = layers.Conv1D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv1d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

