import math
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, LeakyReLU, LayerNormalization
from tensorflow.keras.models import Model


class BSTLayer(tf.keras.layers.Layer):
    def __init__(self, user_count, item_emb_size, cat_emb_size,
                 position_emb_size, act, is_sparse, use_DataLoader, item_count,
                 cat_count, position_count, n_encoder_layers, d_model, d_key,
                 d_value, n_head, dropout_rate, postprocess_cmd,
                 preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                 relu_dropout, layer_sizes):
        super(BSTLayer, self).__init__()

        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.position_emb_size = position_emb_size
        self.act = act
        self.is_sparse = is_sparse
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count
        self.position_count = position_count
        self.user_count = user_count
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.postprocess_cmd = postprocess_cmd
        self.preprocess_cmd = preprocess_cmd
        self.prepostprocess_dropout = prepostprocess_dropout
        self.d_inner_hid = d_inner_hid
        self.relu_dropout = relu_dropout
        self.layer_sizes = layer_sizes

        self.bst = BST(user_count, item_emb_size, cat_emb_size,
                       position_emb_size, act, is_sparse, use_DataLoader,
                       item_count, cat_count, position_count, n_encoder_layers,
                       d_model, d_key, d_value, n_head, dropout_rate,
                       postprocess_cmd, preprocess_cmd, prepostprocess_dropout,
                       d_inner_hid, relu_dropout, layer_sizes)

        self.bias = self.add_weight(
            shape=[1],
            initializer=tf.initializers.Zeros(),
            dtype=tf.float32,
            trainable=True
        )

    def call(self, userid, hist_item_seq, hist_cat_seq, position_seq,
             target_item, target_cat, target_position):
        y_bst = self.bst.call(userid, hist_item_seq, hist_cat_seq,
                              position_seq, target_item, target_cat,
                              target_position)

        predict = tf.sigmoid(y_bst + self.bias)
        return predict


class BST(Model):
    def __init__(self, user_count, item_emb_size, cat_emb_size,
                 position_emb_size, act, is_sparse, use_DataLoader, item_count,
                 cat_count, position_count, n_encoder_layers, d_model, d_key,
                 d_value, n_head, dropout_rate, postprocess_cmd,
                 preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                 relu_dropout, layer_sizes):

        super(BST, self).__init__()
        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.position_emb_size = position_emb_size
        self.act = act
        self.is_sparse = is_sparse
        # significant for speeding up the training process
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count
        self.user_count = user_count
        self.position_count = position_count
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.postprocess_cmd = postprocess_cmd
        self.preprocess_cmd = preprocess_cmd
        self.prepostprocess_dropout = prepostprocess_dropout
        self.d_inner_hid = d_inner_hid
        self.relu_dropout = relu_dropout
        self.layer_sizes = layer_sizes

        init_value_ = 0.1
        self.hist_item_emb_attr = Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ / math.sqrt(float(self.item_emb_size))))

        self.hist_cat_emb_attr = Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ / math.sqrt(float(self.cat_emb_size))))

        self.hist_position_emb_attr = Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ /
                    math.sqrt(float(self.position_emb_size))))

        self.target_item_emb_attr = Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ / math.sqrt(float(self.item_emb_size))))

        self.target_cat_emb_attr = Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ / math.sqrt(float(self.cat_emb_size))))

        self.target_position_emb_attr = Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ /
                    math.sqrt(float(self.position_emb_size))))

        self.userid_attr = Embedding(
            self.user_count,
            self.d_model,
            sparse=False,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                std=init_value_ / math.sqrt(float(self.d_model))))

        self._dnn_layers = []
        sizes = [d_model] + layer_sizes + [1]
        acts = ["relu" for _ in range(len(layer_sizes))] + [None]
        for i in range(len(layer_sizes) + 1):
            linear = Dense(
                units=sizes[i + 1],
                kernel_initializer=tf.keras.initializers.Normal(
                    std=0.1 / math.sqrt(sizes[i])))
            self.add_layer('dnn_linear_%d' % i, linear)
            self._dnn_layers.append(linear)
            if acts[i] == 'elu':
                act = LeakyReLU()
                self.add_layer('dnn_act_%d' % i, act)
                self._dnn_layers.append(act)

        self.drop_out = Dropout(p=dropout_rate)

        self.pff_layer = []
        hid_linear = Dense(
            units=self.d_inner_hid,
            kernel_initializer=tf.keras.initializers.Normal(
                std=0.1 / math.sqrt(self.d_inner_hid)))
        self.add_layer('hid_l', hid_linear)
        self.pff_layer.append(hid_linear)

        m = LeakyReLU()
        self.pff_layer.append(m)
        hid2_linear = Dense(
            units=self.d_model,
            kernel_initializer=tf.keras.initializers.Normal(
                std=0.1 / math.sqrt(self.d_model)))
        self.add_layer('hid2_l', hid2_linear)
        self.pff_layer.append(hid2_linear)

        self.compute_qkv_layer = []
        q_linear = Dense(
            units=d_key * n_head,
            kernel_initializer=tf.keras.initializers.Normal(std=0.1 /
                                                                math.sqrt(d_model)))
        self.add_layer("q_liner", q_linear)
        self.compute_qkv_layer.append(q_linear)

        k_linear = Dense(
            units=d_key * n_head,
            kernel_initializer=tf.keras.initializers.Normal(std=0.1 /
                                                                math.sqrt(d_key)))
        self.add_layer("k_liner", k_linear)
        self.compute_qkv_layer.append(k_linear)

        v_linear = Dense(
            units=d_value * n_head,
            kernel_initializer=tf.keras.initializers.Normal(std=0.1 /
                                                                math.sqrt(d_value)))
        self.add_layer("v_liner", v_linear)
        self.compute_qkv_layer.append(v_linear)

        po_linear = Dense(
            units=d_model,
            kernel_initializer=tf.keras.initializers.Normal(std=0.1 /
                                                                math.sqrt(d_model)))
        self.add_layer("po_liner", po_linear)
        self.compute_qkv_layer.append(po_linear)

    def positionwise_feed_forward(self, x, dropout_rate):
        """
        Position-wise Feed-Forward Networks.
        This module consists of two linear transformations with a ReLU activation
        in between, which is applied to each position separately and identically.
        """
        pff_input = x
        for _layer in self.pff_layer:
            pff_input = _layer(pff_input)

        if dropout_rate:
            pff_input = self.drop_out(pff_input)

        return pff_input

    def pre_post_process_layer_(self,
                                prev_out,
                                out,
                                process_cmd,
                                dropout_rate=0.5):
        """
        Add residual connection, layer normalization and droput to the out tensor
        optionally according to the value of process_cmd.
        This will be used before or after multi-head attention and position-wise
        feed-forward networks.
        """
        out = tf.add(out, prev_out)
        for cmd in process_cmd:
            if cmd == "n":  # add layer normalization
                out = LayerNormalization(
                    axis=len(out.shape) - 1)(out)

            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = self.drop_out(out)
        return out

    def pre_post_process_layer(self, out, process_cmd, dropout_rate=0.5):
        """
        Add residual connection, layer normalization and droput to the out tensor
        optionally according to the value of process_cmd.
        This will be used before or after multi-head attention and position-wise
        feed-forward networks.
        """
        for cmd in process_cmd:
            if cmd == "a":  # add residual connection
                out = out
            elif cmd == "n":  # add layer normalization
                out = LayerNormalization(
                    axis=len(out.shape) - 1)(out)

            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = self.drop_out(out)
        return out

    def multi_head_attention(self, queries, keys, values, d_key, d_value,
                             d_model, n_head, dropout_rate):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        # print(keys.shape)
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        ):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        def __compute_qkv(queries, keys, values):
            """
            Add linear projection to queries, keys, and values.
            """

            q = self.compute_qkv_layer[0](queries)
            k = self.compute_qkv_layer[1](keys)
            v = self.compute_qkv_layer[2](values)
            return q, k, v

        def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
            """
            Reshape input tensors at the last dimension to split multi-heads
            and then transpose. Specifically, transform the input tensor with shape
            [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
            with shape [bs, n_head, max_sequence_length, hidden_dim].
            """
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            reshaped_q = tf.reshape(x=queries, shape=[0, 0, n_head, d_key])
            # permuate the dimensions into:
            # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
            q = tf.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
            # For encoder-decoder attention in inference, insert the ops and vars
            # into global block to use as cache among beam search.
            reshaped_k = tf.reshape(x=keys, shape=[0, 0, n_head, d_key])
            k = tf.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
            reshaped_v = tf.reshape(
                x=values, shape=[0, 0, n_head, d_value])
            v = tf.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

            return q, k, v

        def scaled_dot_product_attention(q, k, v, d_key, dropout_rate):
            """
            Scaled Dot-Product Attention
            """
            product = tf.matmul(x=q, y=k, transpose_y=True)

            weights = tf.nn.softmax(x=product)
            if dropout_rate:
                weights = self.drop_out(weights)
            out = tf.matmul(x=weights, y=v)
            return out

        def __combine_heads(x):
            """
            Transpose and then reshape the last two dimensions of inpunt tensor x
            so that it becomes one dimension, which is reverse to __split_heads.
            """
            if len(x.shape) != 4:
                raise ValueError("Input(x) should be a 4-D Tensor.")

            trans_x = tf.transpose(x, perm=[0, 2, 1, 3])
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            return tf.reshape(
                x=trans_x, shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]])

        q, k, v = __compute_qkv(queries, keys, values)
        q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

        ctx_multiheads = scaled_dot_product_attention(q, k, v, d_model,
                                                      dropout_rate)

        out = __combine_heads(ctx_multiheads)

        proj_out = self.compute_qkv_layer[3](out)

        return proj_out

    def encoder_layer(self, x):

        attention_out = self.multi_head_attention(
            self.pre_post_process_layer(x, self.preprocess_cmd,
                                        self.prepostprocess_dropout), None,
            None, self.d_key, self.d_value, self.d_model, self.n_head,
            self.dropout_rate)
        attn_output = self.pre_post_process_layer_(x, attention_out,
                                                   self.postprocess_cmd,
                                                   self.prepostprocess_dropout)

        ffd_output = self.positionwise_feed_forward(attn_output,
                                                    self.dropout_rate)

        return self.pre_post_process_layer_(attn_output, ffd_output,
                                            self.preprocess_cmd,
                                            self.prepostprocess_dropout)

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):

        user_emb = self.userid_attr(userid)

        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)

        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)

        hist_position_emb = self.hist_position_emb_attr(position_seq)

        target_item_emb = self.target_item_emb_attr(target_item)

        target_cat_emb = self.target_cat_emb_attr(target_cat)

        target_position_emb = self.target_position_emb_attr(target_position)

        item_sequence = tf.concat(
            [hist_item_emb, hist_item_emb, hist_position_emb], axis=2)
        target_sequence = tf.concat(
            [target_item_emb, target_item_emb, target_position_emb], axis=2)

        # print(position_sequence_target.shape)
        whole_embedding = tf.concat(
            [item_sequence, target_sequence], axis=1)
        # print(whole_embedding)
        enc_output = whole_embedding
        '''
        for _ in range(self.n_encoder_layers):
            enc_output = self.encoder_layer(enc_output)
        '''
        enc_output = self.encoder_layer(enc_output)
        enc_output = self.pre_post_process_layer(
            enc_output, self.preprocess_cmd, self.prepostprocess_dropout)
        _concat = tf.concat([user_emb, enc_output], axis=1)
        dnn_input = _concat
        for n_layer in self._dnn_layers:
            dnn_input = n_layer(dnn_input)
        dnn_input = tf.reduce_sum(dnn_input, axis=1)
        return dnn_input
