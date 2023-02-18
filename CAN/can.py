# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2021/5/11|下午 05:27
# @Moto   : Knowledge comes from decomposition

from co_action_unit import co_action_unit, mlp_layer, out_layer, Embedding
from AutoDL.models.common import get_loss
import tensorflow as tf


def can(inputs, params, is_test=False):
    """ todo: baseline dnn
    """
    # todo: params part
    select_slot = params['select_slot']

    loss_type = params.get('loss_type', 'logloss')
    logloss_pos_weight = params.get('logloss_pos_weight', 1.0)
    focal_alpha = params.get('focal_alpha', 0.25)
    focal_gamma = params.get('focal_gamma', 2.0)

    dnn_use_bn = params['dnn_use_bn']
    dnn_dropout = params['dnn_dropout']
    dnn_hidden_units = params['dnn_hidden_units']
    activation_function = params['activation_function']
    dnn_l2 = params['dnn_l2']
    embedding_size = params['embedding_dim']

    weight_size = params['weight_size']
    feature_cross_tuple = params["feature_cross_tuple"]
    co_input_size = params["co_input_size"]
    co_layer_num = params['co_layer_num']
    order = params['order']

    # todo: weight object embedding object
    embed_input = Embedding(select_slot, embedding_size)(inputs)
    _label = inputs['label']
    # 展示下输入的形状
    if not is_test:
        tf.logging.info("=" * 40 + "TRAIN_NET" + "=" * 40)
    else:
        tf.logging.info("=" * 40 + "TEST_NET" + "=" * 40)
    for key, info in embed_input.items():
        tf.logging.info(key + ": " + str(info))
    tf.logging.info("=" * 80)

    # embed_input 转换
    parameter_lookup = {}
    embedding_lookup = {}
    for slot in params['slot_sort']:
        if select_slot[slot]['dtype'] == 'dense':
            # arr = tf.split(tf.layers.flatten(embed_input[slot]),
            #                [weight_size, 1], axis=-1)
            # print("=======", embed_input[slot])
            # arr = tf.slice(tf.layers.flatten(
            #     embed_input[slot]), [0, embedding_size-weight_size], [-1, -1])
            embedding_lookup[slot] = embed_input[slot]
        else:
            arr = tf.split(tf.layers.flatten(embed_input[slot]),
                           [weight_size, embedding_size - weight_size], axis=-1)
            parameter_lookup[slot] = arr[0]
            embedding_lookup[slot] = arr[1]

    tf.logging.info("parameter_lookup", parameter_lookup)
    tf.logging.info("embedding_lookup", embedding_lookup)
    # =========================================================================

    # todo: build co-action
    co_action_out_ = []
    for cross_feature in feature_cross_tuple:
        p1, p2 = cross_feature.split("&")
        p_input = tf.slice(parameter_lookup[p2], [0, 0], [-1, co_input_size])
        co_action_out_.append(co_action_unit(
            p1 + "_" + p2, p_input, parameter_lookup[p1],
            co_layer_num, order, activation_function))

    # todo: build dien object
    dien_input_ = []
    for slot in params['slot_sort']:
        dien_input_.append(tf.layers.flatten(embedding_lookup[slot]))
    dien_input_ = tf.concat(dien_input_, -1)
    if dnn_use_bn:
        dien_input_ = tf.contrib.layers.batch_norm(dien_input_, is_training=not is_test)
    if dnn_dropout > 0 and not is_test:
        dien_input_ = tf.nn.dropout(dien_input_, rate=dnn_dropout)
    dien_out_ = mlp_layer(
        'dien_hidden', dien_input_, dnn_hidden_units,
        l2=dnn_l2, use_bn=dnn_use_bn,
        use_bias=True, drop_rate=dnn_dropout, activation=activation_function)

    # concat dien and co-action
    out_ = tf.concat(co_action_out_ + [dien_out_], axis=-1)
    # todo: build mlp out layer
    _input = out_
    if dnn_use_bn:
        _input = tf.contrib.layers.batch_norm(_input, is_training=not is_test)
    if dnn_dropout > 0 and not is_test:
        _input = tf.nn.dropout(_input, rate=dnn_dropout)
    out = mlp_layer(
        'dnn_hidden', _input, dnn_hidden_units, l2=dnn_l2, use_bn=dnn_use_bn,
        use_bias=True, drop_rate=dnn_dropout, activation=activation_function)

    logits = out_layer('out', out, [1], l2=dnn_l2, activation=None)
    preds = tf.sigmoid(logits, name='y')
    tf.logging.info("preds: " + str(preds))
    # todo: loss
    labels = tf.cast(_label, dtype=tf.float32)  # labels：必须是[batch, 1]
    loss_args = {
        'loss_type': loss_type,
        'logloss_pos_weight': logloss_pos_weight,
        'focal_alpha': focal_alpha,
        'focal_gamma': focal_gamma
    }
    _, loss = get_loss(labels, logits, preds, **loss_args)

    if is_test:
        tf.add_to_collections("input_tensor", embed_input)
        tf.add_to_collections("output_tensor", preds)
    net_dic = {
        "loss": loss,
        "ground_truth": inputs["label"][:, 0],
        "prediction": preds}

    return net_dic
