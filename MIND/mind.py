# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/3/4|下午 07:24
# @Moto   : Knowledge comes from decomposition
from autoMLv2.models.common import activation_fn, get_optimizer, get_train_op
from autoMLv2.layers import nn_tower, caps_with_mind_layer, \
    label_aware_attention
from autoMLv2.libs.features import build_feature_columns
import tensorflow as tf
import collections


def mind(features, labels, mode, params):
    """
    todo: 采用动态路由的机制来挖掘用户的多层次兴趣，丰富对于用户兴趣的表达
        对于不同的用户兴趣采用label-aware attention layer来归纳兴趣的偏好
        最为核心的就是精准的捕捉用户的兴趣，为用户推荐他们真正感兴趣的内容
        对于单个用户采用多个向量表征其行为特征（label-aware attention）；在召回阶段提出MIND模型来捕捉用户不同方面的兴趣（capsule routing）
        线上系统通常会将请求的相关信息存储到日志中，日志中的每条样本主要包括以下的一些基本信息，（1）与用户产生交互的商品信息（2）用户本身的一些信息（如用户年龄、性别等）（3）目标商品的相关信息（如商品类别、商品id等）
    MIND模型的核心任务就是学习一个映射函数，该函数将将原始的特征映射为用户表征向量
    MIND模型的输入为用户相关的特征，输出是多个用户表征向量，主要用于召回阶段使用
    流程： 输入的特征主要包括三部分，即用户自身相关特征、用户行为特征（如浏览过的商品id）和label特征。所有的id类特征都经过Embedding层，其中对于用户行为特征对应的Embedding向量进行average pooling操作，然后将用户行为Embedidng向量传递给Multi-Interest Extract layer（生成interest capsules），将生成的interest capsules与用户本身的Embedding concate起来经过几层全连接网络，就得到了多个用户表征向量，在模型的最后有一个label-aware attention层。在线上使用的时候同样应该采用类似向量化召回的思路，选取TopN

    Embedding&Pooling Layer
    Multi-interest Extract Layer

    训练样本是 [user_profile, read_item1_profile, read_item2_profile, next_item_profile]
    文章指出原始的动态路由无法直接应用到MIND模型中，所以文章主要做了以下几方面的改进：（1）不同行为/兴趣capsule之间共享双线性映射网络S（2）随机初始化routing logit（3）自适应调整用户兴趣表征向量的个数

    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    # todo: get use params

    # caps
    batch_size = params['batch_size']
    iter_routing = params['iter_routing']
    out_units = params['out_units']
    seq_max_len = params['seq_max_len']
    interest_max_num = params['interest_max_num']
    # net
    batch_norm = params['batch_norm']
    dnn_dropout = params['dnn_dropout']
    dnn_hidden_units = params['dnn_hidden_units']
    activation_function = activation_fn(params['activation_function'])
    dnn_l2 = params['dnn_l2']
    embedding_dim = params['embedding_dim']
    ext_sampled_num = params['ext_sampled_num']

    # feature
    select_features = params['select_features_config']
    user_features = params['user_features']
    item_features = params['item_features']
    features_config = params['features_config']
    cross_features_config = params['cross_features_config']

    # optimizer
    optimizer_type = params['optimizer']
    optimizer_lr = params['optimizer_lr']
    optimizer_lr_decay_rate = params['optimizer_lr_decay_rate']
    optimizer_lr_decay_steps = params['optimizer_lr_decay_steps']
    optimizer_adam_beta1 = params['optimizer_adam_beta1']
    optimizer_adam_beta2 = params['optimizer_adam_beta2']
    optimizer_ftrl_l1 = params['optimizer_ftrl_l1']
    optimizer_ftrl_l2 = params['optimizer_ftrl_l2']
    optimizer_ftrl_beta = params['optimizer_ftrl_beta']

    # todo: build feature columns object
    feature_columns_ob = build_feature_columns(
        select_features, features_config, cross_features_config,
        embedding_dim=embedding_dim)
    # todo: group feature, 特征分成3部分：用户特征，序列特征（每一个序列特征包含其自带所有特征）
    #  这里务必注意： features的格式：
    #    { "label": [1] 固定值1
    #      "user_id":
    #      "user_cate":
    #      # 物品特征
    #      item_id:
    #      item_cate:
    #      # 行为特征
    #      "seq_len":  # 该特征必备，表示行为的实际长度
    #      "item_id_0":
    #      "item_id_1":   # 这个是行为特征，有最大特征数max_len
    #      "item_cate_0":
    #      "item_cate_1": # 这个是行为特征，有最大特征数max_len
    #      # 其中物品特征和行为特征，由于都是用的物品的特征，因此这里是共用原子特征（item_id和item_cate）
    #

    # 抽取出用户, 物品的feature_columns_ob
    #
    user_columns_ob = {k: v for k, v in feature_columns_ob.items()
                       if k in user_features}
    item_columns_ob = {k: v for k, v in feature_columns_ob.items()
                       if k in item_features}
    if len(features['seq_len'].get_shape()) <= 1:
        seq_len = tf.expand_dims(
            tf.cast(features['seq_len'], tf.int32), axis=-1)
    else:
        seq_len = tf.cast(features['seq_len'], tf.int32)

    def get_input_embedding(features, columns_ob):
        _input = []
        for k, v in columns_ob.items():
            feed = tf.feature_column.input_layer(
                features, v.get_feature_column())
            if features_config[k]['type'] == 'continuous':
                weight = tf.get_variable(
                    name=k + '_v', shape=[1, embedding_dim],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                feed = tf.contrib.layers.batch_norm(
                    feed, is_training=is_training)
                feed = tf.matmul(feed, weight)
            _input.append(tf.expand_dims(feed, axis=1))
        _input = tf.concat(_input, axis=1)
        return _input  # [batch, feature, embedding]

    # todo: user profile 也就是论文中的other feature
    user_features_embedding = get_input_embedding(
        features, user_columns_ob)  # [batch, f, embedding]
    user_other_embedding = tf.layers.flatten(
        user_features_embedding)  # [batch, f*embedding]

    # todo: item profile 也就是论文里的 label
    item_feature_embedding = get_input_embedding(
        features, item_columns_ob)
    items_embedding = tf.reduce_mean(
        item_feature_embedding, axis=1)  # [batch, embedding]

    # todo: user behavior 也就是论文中[item, item, item] list
    behavior_embedding = []
    for i in range(seq_max_len):
        newf = collections.OrderedDict()
        for item in item_features:
            key = item + "_" + str(i)
            newf[item] = features[key]
        behavior_ = tf.reduce_mean(
            get_input_embedding(newf, item_columns_ob), axis=1)
        behavior_embedding.append(tf.expand_dims(behavior_, axis=1))
    behavior_embedding = tf.concat(behavior_embedding,
                                   axis=1)  # [batch, max_len, embedding]

    print(user_features_embedding)
    print(item_feature_embedding)
    print(behavior_embedding)

    high_capsule = caps_with_mind_layer(
        behavior_embedding, seq_len, out_units, seq_max_len, interest_max_num,
        iter_routing)  # [batch, interest_max_num, out_units]

    # 组合user_features_embedding 和 user_hist_item_capsule
    user_other_embedding = tf.tile(tf.expand_dims(
        user_other_embedding, 1),
        [1, interest_max_num, 1])  # [batch, interest_max_num, f*embedding]

    h_layer_input_ = tf.concat(
        [user_other_embedding, high_capsule],
        axis=2)  # [batch, interest_max_num, out_units+f*embedding]

    if dnn_dropout > 0 and is_training:
        h_layer_input_ = tf.nn.dropout(h_layer_input_, rate=dnn_dropout)
    # 注意： dnn_hidden_units[-1] == embedding_dim
    assert dnn_hidden_units[-1] == embedding_dim
    if batch_norm:
        h_layer_input_ = tf.contrib.layers.batch_norm(
            h_layer_input_, is_training=is_training)
    user_embedding = nn_tower(
        'user_dnn_hidden',
        h_layer_input_, dnn_hidden_units,
        use_bias=True, activation=activation_function,
        l2=dnn_l2
    )  # [batch, interest_max_num, dnn_hidden_units[-1]]

    # 加入attention机制
    user_out_ = label_aware_attention(
        user_embedding, items_embedding, user_embedding, seq_len,
        pow_p=1.0, k_max=interest_max_num,
        dynamic_k=False)  # [batch, interest_max_num, dnn_hidden_units[-1]]

    # todo: model train and predict config
    if mode == tf.estimator.ModeKeys.PREDICT:
        # logits = tf.matmul(user_out_, tf.transpose(items_embedding))
        # classes = tf.argmax(logits, axis=2)  # 获得每个兴趣的最大item得分的下标
        # classes = tf.argmax(tf.argmax(logits, axis=2))  # 获得所有兴趣的最大item得分的下标
        predictions = {
            'items_embedding': items_embedding,
            'user_embedding': user_out_,
        }
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(
                predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions,
            export_outputs=export_outputs,
        )
    else:
        # todo: softmax 计算item数量的每个分类的分值，
        #  其中优化点在于对负样本随机抽取一定的量进行计算loss
        #  故loss为正样本和随机的N个负样本的loss综合
        #  这里一定要理解tf.nn.sampled_softmax_loss 将多分类问题转换成多个二分类，从而获得loss

        zero_bias = tf.get_variable(
            "bias", shape=[batch_size],
            initializer=tf.zeros_initializer,
            trainable=False)
        # todo: 【这里是一个隐患】，留意下，我这里是基于batch生成一个下标索引，就是怕输入的batch数据要是没有batch大就有问题
        #   注意这个与youtubeDNN有点不一样，这里只采用在batch内进行采样本，而youtube是在整个候选矩阵中进行采样
        item_index = tf.expand_dims(
            tf.constant(list(range(batch_size))), axis=-1)
        # batch_size_tmp = tf.shape(items_embedding)[0]
        # item_index = tf.expand_dims(
        #     tf.range(tf.shape(items_embedding)[0]), axis=-1)
        losses = tf.nn.sampled_softmax_loss(
            weights=items_embedding,
            biases=zero_bias,
            labels=item_index,
            inputs=user_out_,
            num_sampled=ext_sampled_num,
            num_classes=batch_size
        )
        tf.summary.histogram('loss', losses)
        loss = tf.reduce_sum(losses)
        optimizer_args = {
            'optimizer_lr': optimizer_lr,
            'optimizer_lr_decay_rate': optimizer_lr_decay_rate,
            'optimizer_lr_decay_steps': optimizer_lr_decay_steps,
            'optimizer_adam_beta1': optimizer_adam_beta1,
            'optimizer_adam_beta2': optimizer_adam_beta2,
            'optimizer_ftrl_l1': optimizer_ftrl_l1,
            'optimizer_ftrl_l2': optimizer_ftrl_l2,
            'optimizer_ftrl_beta': optimizer_ftrl_beta,
        }
        optimizer = get_optimizer(optimizer_type, **optimizer_args)
        train_op = get_train_op(optimizer.minimize(loss))

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss,
                train_op=train_op,
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss,
            )
