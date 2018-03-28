import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import initializers
from keras import regularizers
from keras import constraints

def axes_dot(x, y, axes):
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = K.ndim(x)
    y_ndim = K.ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    
    if axes[0] == axes[1]:
        out = tf.reduce_sum(tf.multiply(x, y), axes[0])
    else:
        out = tf.reduce_sum(tf.multiply(x, tf.transpose(y, [axes[1], axes[0]])), axes[0])
    
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if K.ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

class Attention(Layer):
    def __init__(self, input_units,
                 input_metrix=False,
                 input_square=False,
                 add_square=False,
                 kernel_initializer='uniform', 
                 kernel_regularizer=None, 
                 kernel_constraint=None, 
                 **kwargs):
        self.input_units=input_units
        self.input_metrix=input_metrix
        self.input_square=input_square
        self.add_square=add_square
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert len(input_shape) >= 2
        if self.add_square:
            input_len = input_shape[-1] * 2
        else:
            input_len = input_shape[-1]
        
        if self.input_metrix:
            self.kernel = self.add_weight(name='kernel', 
                                          shape=(input_len,input_len),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        else:
            self.kernel = self.add_weight(name='kernel', 
                                          shape=(input_len,1),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)

        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x):
        if self.add_square:
            y = K.concatenate([x, K.square(x)], axis=-1)
        elif self.input_square:
            y = K.square(x)
        else:
            y = x
        
        if self.input_metrix:
            match = axes_dot(y, K.dot(y, self.kernel), axes=[2,2])
        else:
            match = K.dot(y, self.kernel)
        
        match = K.softmax(match)
        return K.batch_dot(x[:,:,:self.input_units], match, axes=[1,1])
    
    def compute_output_shape(self, input_shape):
        return input_shape[:1]+tuple([self.input_units])