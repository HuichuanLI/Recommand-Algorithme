import tensorflow as tf
from tensorflow.keras import backend as K
#from tensorflow.python.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM

## 自定义层
class AttentionLayer(Layer):
    def __init__(self, hidden_size=hidden_size**kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
        
    ##build函数 创建weights使用    
    def build(self, input_shape):
        assert len(input_shape) == 3  ## [batch_size, maxlen, hidden_size]
        
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
            
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    ##call函数 前项运算(类比于pytorch的forword函数)
    def call(self, inputs):
        self.V = tf.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1) ## 加权平均
        return outputs


class TextAttBiRNN(Model):
    def __init__(self, maxlen, max_features, embedding_dims, attention_size
                 class_num=3,
                 last_activation='softmax',
                 hidden_size):

        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.attention_size = attention_size
        self.hidden_size = hidden_size

    def get_model(self):
        input = Input((self.maxlen,))

        ##创建Embedding层
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input) 
        ##创建RNN层
        x = Bidirectional(LSTM(hidden_size, return_sequences=True))(embedding)  # LSTM or GRU
        ##创建Attention层
        x = AttentionLayer(self.attention_size, self.hidden_size)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


maxlen = 'your maxlen' ## 文本长度
max_features = 'your XXX' + 1 ## 字典大小
attention_size = 'your attention_size' ## attention输出vector的大小
epochs = 'your epochs'
batch_size = 'your batch_size'
hidden_size = 128

model = TextAttBiRNN(maxlen, max_features, embedding_dims, attention_size,hidden_size).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs)


