# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/3/3|下午 07:56
# @Moto   : Knowledge comes from decomposition

import tensorflow as tf
import numpy as np


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return shape


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(
        vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise

    return vec_squashed


def routing(input, b_IJ, iter_routing, num_outputs=10, num_dims=16):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
        num_outputs: the number of output capsules.
        num_dims: the number of dimensions for output capsule.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    input_shape = get_shape(input)
    W = tf.get_variable(
        'Weight',
        shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.01))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
    # assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat,
                       shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    # assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3,
                                            keepdims=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return v_J


def caps_layer(
        input_, num_outputs, vec_len, batch_size, iter_routing,
        kernel_size=None, stride=None,
        with_routing=True, layer_type='FC'):
    """
    The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
    """
    if layer_type == 'CONV':
        if not with_routing:
            # the PrimaryCaps layer, a convolutional layer
            # input: [batch_size, 20, 20, 256]
            # assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

            # NOTE: I can't find out any words from the paper whether the
            # PrimaryCap convolution does a ReLU activation or not before
            # squashing function, but experiment show that using ReLU get a
            # higher test accuracy. So, which one to use will be your choice
            capsules = tf.layers.conv2d(
                input_,
                num_outputs * vec_len,
                kernel_size, stride,
                padding="VALID",
                activation_fn=tf.nn.relu)

            capsules = tf.reshape(capsules, (batch_size, -1, vec_len, 1))
            capsules = squash(capsules)
            return capsules

    if layer_type == 'FC':
        if with_routing:
            # the DigitCaps layer, a fully connected layer
            # Reshape the input into [batch_size, 1152, 1, 8, 1]
            input_ = tf.reshape(input_, shape=(
                batch_size, -1, 1, input_.shape[-2].value, 1))
            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros(
                    [batch_size, input_.shape[1].value, num_outputs, 1, 1],
                    dtype=np.float32))
                capsules = routing(input_, b_IJ, iter_routing,
                                   num_outputs=num_outputs,
                                   num_dims=vec_len)
                capsules = tf.squeeze(capsules, axis=1)
            return capsules


def squash_with_mind(vector):
    """todo: 类似于神经网络里的激活函数， 也就是对应原始的数据进行了一个压缩
             这里完全复原论文的公式： S*(|S|^2 / (1+|S|^2) / |s|)
             输入：
                [batch_size, num_caps, vec_len]
             输出：
                [batch_size, num_caps, vec_len]
    """
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(
        vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise

    return vec_squashed


def caps_with_mind_layer(
        behavior_embeddings, seq_len,
        out_units, max_len, k_max, iter_routing):
    """ todo: 胶囊网络的核心公式
            S = C1 * W1 * V1 + C2 * W2 * V2
            V_out_ = squash(S)
            V1 和 V2 是两个向量， 这里类比于神经元的X1和X2
            W1 和 W2 是初始化权重，这里类比于神经元的权重w1和w2
            C1 和 C2 是动态权重，这个在神经元中没有，这里独有，而且他不能进行反向传播进行更新
         除开C1 和 C2 会发现与神经元很像
         -------------------------------------
         这里开始重点介绍C1和C2如何进行确定和更新（不是通过反向传播进行更新的）
         初始化 b1 = 0 ， b2 = 0, u1 = w1*v1 , u2 = w2*v2
         for r = 1 to T do
            c1, c2 = softmax(b1, b2)  # 这里是确保C1+C2 = 1
            S = c1*u1 + c2*u2
            a = Squash(S)
            b1 = b1 + a*u1  # 开始更新
            b2 = b2 + a*u2

         ----------------------------------------------------------
         behavior_embeddings -> [batch, seq, embedding]
         seq_len， max_len
         input_units = embedding_dim
         out_units 输出后的embedding长度
         k_max  兴趣的最大数
         iter_routing： 网络迭代多少次，一般为3次
    """
    seq_len_tile = tf.tile(seq_len, [1, k_max])
    batch_size = tf.shape(behavior_embeddings)[0]
    embedding_dim = behavior_embeddings.get_shape().as_list()[-1]

    B = tf.get_variable(
        shape=[1, k_max, max_len],
        initializer=tf.truncated_normal_initializer(stddev=0.01),
        trainable=False, name="B",
        dtype=tf.float32)
    W = tf.get_variable(
        shape=[embedding_dim, out_units],
        initializer=tf.truncated_normal_initializer(stddev=0.01),
        name="S", dtype=tf.float32)

    for i in range(iter_routing):
        # 对序列进行mask，mask的会设置成一个很小的值-2 ** 32 + 1
        mask = tf.sequence_mask(seq_len_tile, max_len)
        pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
        B_with_padding = tf.where(
            mask,
            tf.tile(B, [batch_size, 1, 1]), pad)  # [batch, k_max, max_len]

        C = tf.nn.softmax(B_with_padding)  # [batch, k_max, max_len]
        # behavior_embeddings -> [batch, max_len, embedding_dim]
        # weight -> [embedding_dim, out_units]
        U = tf.tensordot(
            behavior_embeddings, W, axes=1)  # [batch, max_len, out_units]

        S = tf.matmul(C, U)  # [batch, k_max, out_units]
        # 压缩
        A = squash_with_mind(S)  # [batch, k_max, out_units]

        # 开始更新B
        delta_B = tf.reduce_sum(
            tf.matmul(A, tf.transpose(U, perm=[0, 2, 1])),
            axis=0, keep_dims=True
        )  # [batch, k_max, max_len]

        B = tf.add(B, delta_B)

    A = tf.reshape(A, [-1, k_max, out_units])
    return A


def label_aware_attention(key, query, value, seq_len, pow_p, k_max, dynamic_k):
    # matmul

    weight = tf.reduce_sum(
        key * tf.tile(tf.expand_dims(query, 1), [1, k_max, 1]),
        axis=-1, keep_dims=True)
    # power
    weight = tf.pow(weight, pow_p)  # [x,k_max,1]

    # 自适应调整用户兴趣表征向量的个数
    if dynamic_k:
        k_user = tf.cast(tf.maximum(
            1.,
            tf.minimum(
                tf.cast(k_max, dtype="float32"),  # k_max
                tf.log1p(tf.cast(seq_len, dtype="float32")) / tf.log(2.)
                # hist_len
            )
        ), dtype="int64")
        seq_mask = tf.transpose(tf.sequence_mask(k_user, k_max), [0, 2, 1])
        padding = tf.ones_like(seq_mask, dtype=tf.float32) * (
                -2 ** 32 + 1)  # [x,k_max,1]
        weight = tf.where(seq_mask, weight, padding)

    #
    weight = tf.nn.softmax(weight, dim=1, name="weight")
    output = tf.reduce_sum(value * weight, axis=1)
    return output
