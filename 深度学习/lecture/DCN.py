import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense

class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layer = [Dense(x, activation=activation) for x in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output

class Cross_layer(Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super().__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w'+str(i),
                            shape=(input_shape[1], 1), # 跟输入维度相同的向量
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)]            # 每层对应不同的w
        self.cross_bias = [
            self.add_weight(name='b'+str(i),
                            shape=(input_shape[1], 1), # 跟输入维度相同的向量
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)]            # 每层对应不同的b

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)  # (None, dim, 1)
        xl = x0                              # (None, dim, 1)
        for i in range(self.layer_num):
            # 先乘后两项得到标量，便于计算
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i]) # (None, 1, 1)
            # 再乘上x0，加上b、xl
            xl = tf.matmul(x0, xl_w) + self.cross_bias[i] + xl  # (None, dim, 1)

        output = tf.squeeze(xl, axis=2)  # (None, dim)
        return output


from layer import Dense_layer, Cross_layer

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras import Model

class DCN(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, layer_num, reg_w=1e-4, reg_b=1e-4):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
             for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dense_layer = Dense_layer(hidden_units, output_dim, activation)
        self.cross_layer = Cross_layer(layer_num, reg_w=reg_w, reg_b=reg_b)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=1)

        # Crossing layer
        cross_output = self.cross_layer(x)

        # Dense layer
        dnn_output = self.dense_layer(x)

        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))
        return output

