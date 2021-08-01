import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense

class FM_layer(Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), # [field,1]
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k), # [field, k]
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)

        sum_pow = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        pow_sum = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(sum_pow - pow_sum, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return output

class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation

        self.hidden_layer = [Dense(i, activation=self.activation) for i in self.hidden_units]
        self.output_layer = Dense(self.output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


Model 搭建：

from layer import FM_layer, Dense_layer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding

class DeepFM(Model):
    def __init__(self, feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # 将类别特征从onehot_dim转换成embed_dim
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
             for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.FM = FM_layer(k, w_reg, v_reg)
        self.Dense = Dense_layer(hidden_units, output_dim, activation)


    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        fm_output = self.FM(x)
        dense_output = self.Dense(x)
        # output = tf.nn.sigmoid((fm_output + dense_output))
        output = tf.concat([fm_output, dense_output], axis = -1)
        output = Dense(1, act = 'sigmoid')(output)
        return output

