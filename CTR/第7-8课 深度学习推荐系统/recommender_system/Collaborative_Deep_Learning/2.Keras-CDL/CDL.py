import numpy as np
import logging
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout, Lambda, Add
from keras.layers.noise import GaussianNoise
from keras.initializers import RandomUniform, RandomNormal
from keras.models import Model
from keras.regularizers import l2
from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer

class CollaborativeDeepLearning:
    def __init__(self, item_mat, hidden_layers):
        '''
        hidden_layers = a list of three integer indicating the embedding dimension of autoencoder
        item_mat = item feature matrix with shape (# of item, # of item features)
        '''
        assert(len(hidden_layers)==3)
        self.item_mat = item_mat
        self.hidden_layers = hidden_layers
        self.item_dim = hidden_layers[0]
        self.embedding_dim = hidden_layers[-1]
        
    def pretrain(self, lamda_w=0.1, encoder_noise=0.1, dropout_rate=0.1, activation='sigmoid', batch_size=64, epochs=10):
        '''
        layer-wise pretraining on item features (item_mat)
        '''
        self.trained_encoders = []
        self.trained_decoders = []
        X_train = self.item_mat
        for input_dim, hidden_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            logging.info('Pretraining the layer: Input dim {} -> Output dim {}'.format(input_dim, hidden_dim))
            pretrain_input = Input(shape=(input_dim,))
            encoded = GaussianNoise(stddev=encoder_noise)(pretrain_input)
            encoded = Dropout(dropout_rate)(encoded)
            encoder = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(lamda_w), bias_regularizer=l2(lamda_w))(encoded)
            decoder = Dense(input_dim, activation=activation, kernel_regularizer=l2(lamda_w), bias_regularizer=l2(lamda_w))(encoder)
            # autoencoder
            ae = Model(inputs=pretrain_input, outputs=decoder)
            # encoder
            ae_encoder = Model(inputs=pretrain_input, outputs=encoder)
            # decoder
            encoded_input = Input(shape=(hidden_dim,))
            decoder_layer = ae.layers[-1] # the last layer
            ae_decoder = Model(encoded_input, decoder_layer(encoded_input))

            ae.compile(loss='mse', optimizer='rmsprop')
            ae.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=2)

            self.trained_encoders.append(ae_encoder)
            self.trained_decoders.append(ae_decoder)
            X_train = ae_encoder.predict(X_train)

    def fineture(self, train_mat, test_mat, lamda_u=0.1, lamda_v=0.1, lamda_n=0.1, lr=0.001, batch_size=64, epochs=10):
        '''
        Fine-tuning with rating prediction
        '''
        num_user = int( max(train_mat[:,0].max(), test_mat[:,0].max()) + 1 )
        num_item = int( max(train_mat[:,1].max(), test_mat[:,1].max()) + 1 )

        # item autoencoder 
        itemfeat_InputLayer = Input(shape=(self.item_dim,), name='item_feat_input')
        encoded = self.trained_encoders[0](itemfeat_InputLayer)
        encoded = self.trained_encoders[1](encoded)
        decoded = self.trained_decoders[1](encoded)
        decoded = self.trained_decoders[0](decoded)

        # user embedding
        user_InputLayer = Input(shape=(1,), dtype='int32', name='user_input')
        user_EmbeddingLayer = Embedding(input_dim=num_user, output_dim=self.embedding_dim, input_length=1, name='user_embedding', embeddings_regularizer=l2(lamda_u), embeddings_initializer=RandomNormal(mean=0, stddev=1))(user_InputLayer)
        user_EmbeddingLayer = Flatten(name='user_flatten')(user_EmbeddingLayer)

        # item embedding
        item_InputLayer = Input(shape=(1,), dtype='int32', name='item_input')
        item_OffsetVector = Embedding(input_dim=num_item, output_dim=self.embedding_dim, input_length=1, name='item_offset_vector', embeddings_regularizer=l2(lamda_v), embeddings_initializer=RandomNormal(mean=0, stddev=1))(item_InputLayer)
        item_OffsetVector = Flatten(name='item_flatten')(item_OffsetVector)
        item_EmbeddingLayer = Add()([encoded, item_OffsetVector]) 
        
        # rating prediction
        dotLayer = Dot(axes = -1, name='dot_layer')([user_EmbeddingLayer, item_EmbeddingLayer])

        self.cdl_model = Model(inputs=[user_InputLayer, item_InputLayer, itemfeat_InputLayer], outputs=[dotLayer, decoded])
        self.cdl_model.compile(optimizer='rmsprop', loss=['mse', 'mse'], loss_weights=[1, lamda_n])

        train_user, train_item, train_item_feat, train_label = self.matrix2input(train_mat)
        test_user, test_item, test_item_feat, test_label = self.matrix2input(test_mat)

        model_history = self.cdl_model.fit([train_user, train_item, train_item_feat], [train_label, train_item_feat], epochs=epochs, batch_size=batch_size, validation_data=([test_user, test_item, test_item_feat], [test_label, test_item_feat]))
        return model_history

        # v and theta
        '''
        def lossLayer(args):
            Vj, Thetaj = args
            return 0.5 * K.mean(K.square(Vj - Thetaj), axis=1)
        
        
        class lossLayer(Layer):
            def __init__(self, **kwargs):
                super(lossLayer, self).__init__(**kwargs)
                #self.kernel_regularizer = l2(lamda_v)

            def call(self, inputs):
                Vj, Thetaj = inputs
                return 0.5 * K.mean(K.square(Vj - Thetaj), axis=1)

            def compute_output_shape(self, input_shape):
                return (None, 1)
        
        def fe_loss(y_true, y_pred):
            return y_pred

        fe_regLayer = lossLayer()([encoded, item_EmbeddingLayer])
        

        my_RMSprop = optimizers.RMSprop(lr=lr)

        self.cdl_model = Model(inputs=[user_InputLayer, item_InputLayer, itemfeat_InputLayer], outputs=[dotLayer, decoded, fe_regLayer])
        self.cdl_model.compile(optimizer='rmsprop', loss=['mse', 'mse', fe_loss], loss_weights=[1, lamda_n, lamda_v])

        train_user, train_item, train_item_feat, train_label = self.matrix2input(train_mat)
        test_user, test_item, test_item_feat, test_label = self.matrix2input(test_mat)

        model_history = self.cdl_model.fit([train_user, train_item, train_item_feat], [train_label, train_item_feat, train_label], epochs=epochs, batch_size=batch_size, validation_data=([test_user, test_item, test_item_feat], [test_label, test_item_feat, test_label]))
        return model_history
        '''

    def matrix2input(self, rating_mat):
        train_user = rating_mat[:, 0].reshape(-1, 1).astype(int)
        train_item = rating_mat[:, 1].reshape(-1, 1).astype(int)
        train_label = rating_mat[:, 2].reshape(-1, 1)
        train_item_feat = [self.item_mat[train_item[x]][0] for x in range(train_item.shape[0])]
        return train_user, train_item, np.array(train_item_feat), train_label
    
    def build(self):
        # rating prediction
        prediction_layer = Dot(axes = -1, name='prediction_layer')([user_EmbeddingLayer, encoded])
        self.model = Model(inputs=[user_InputLayer, itemfeat_InputLayer], outputs=[prediction_layer])
        
    def getRMSE(self, test_mat):
        test_user, test_item, test_item_feat, test_label = self.matrix2input(test_mat)
        pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        # pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        return np.sqrt(np.mean(np.square(test_label.flatten() - pred_out[0].flatten())))

'''
from keras.engine.topology import Layer
class V_Theta_Layer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='epsilon', 
                                        shape=(input_shape[1], ),
                                        initializer=RandomNormal(mean=0., stddev=lamda_v),
                                        regularizer=l2(lamda_v),
                                        trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel
'''