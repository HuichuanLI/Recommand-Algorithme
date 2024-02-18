# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/10/13|11:11
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from .layer import mlp_layer, out_layer


# 多层渐进式的ple,是cgc网络的基础上进行渐进式多层cgc网络
def ple_model(inputs, conf, is_test=False):
    # =================================================
    # 统一模型input部分
    # =================================================
    dense_embed = tf.gather(inputs["ps_embed"], inputs["dense_id"])  # batch * dense_feature_n * embed
    sparse_embed = tf.reshape(
        tf.math.segment_sum(
            tf.gather(inputs["ps_embed"], inputs["sparse_id"]), inputs["sparse_indice"]),
        [-1, conf.multi_hot_slot_num, conf.embed_dim])  # batch * sparse_feature_n * embed
    embed_input = tf.concat([dense_embed, sparse_embed], axis=1)  # # batch * feature_n * embed

    # =================================================
    # 模型核心
    # =================================================

    # dnn
    deep_layer_dim = conf.deep_layer_dim
    use_bn = conf.use_bn
    bn_decay = conf.bn_decay
    bn_epsilon = conf.bn_epsilon
    use_drop = conf.use_drop
    drop_rate = conf.drop_rate
    deep_act = conf.deep_act
    expert_out_size = conf.expert_out_size
    # 与mmoe不同的地方：区分专家的性质，分为专属专家和共有专家
    ple_num = conf.ple_num
    exclusive_expert_num = conf.exclusive_expert_num
    common_expert_num = conf.common_expert_num
    target_num = conf.target_num
    target1_wgt = conf.target1_wgt
    target2_wgt = conf.target2_wgt
    # 专属expert是均分给每个目标
    exclusive_expert_layers = [expert_out_size] * exclusive_expert_num  # 每个专家的输出都是expert_out_size
    common_expert_layers = [expert_out_size] * common_expert_num  # 共有的输出都是expert_out_size
    # embed_input reshape
    embed_input_shape = embed_input.get_shape().as_list()
    embed_input_reshape = tf.reshape(
        embed_input, shape=[-1, embed_input_shape[1] * embed_input_shape[2]])

    # ----- build exclusive_expert -----
    # exclusive_expert 的输出大小必须保持一致，后面需要对每个expert 进行加权求和
    with tf.variable_scope("Exclusive-Experts—{}".format(0)):
        exclusive_experts = []
        for i, layer in enumerate(exclusive_expert_layers):
            expert_out = mlp_layer(
                name_scope="basic", inputs=embed_input_reshape, units=[layer],
                use_bn=use_bn, bn_decay=bn_decay, bn_epsilon=bn_epsilon,
                use_drop=use_drop, drop_rate=drop_rate,
                activation=["relu"], is_training=not is_test)
            exclusive_experts.append(expert_out)  # list([batch, expert_out_size]*exclusive_expert_num)

    # ----- build common_expert -----
    # common_expert 的输出大小必须保持一致，后面需要对每个expert 进行加权求和
    with tf.variable_scope("Common-Experts—{}".format(0)):
        common_experts = []
        for i, layer in enumerate(common_expert_layers):
            expert_out = mlp_layer(
                name_scope="basic", inputs=embed_input_reshape, units=[layer],
                use_bn=use_bn, bn_decay=bn_decay, bn_epsilon=bn_epsilon,
                use_drop=use_drop, drop_rate=drop_rate,
                activation=["relu"], is_training=not is_test)
            common_experts.append(expert_out)  # list([batch, expert_out_size]*common_expert_num)
    # 新增多层渐进
    for ple in range(ple_num):
        # ----- build Gates -----
        with tf.variable_scope("Gate-Weight-{}".format(i)):
            # 基于输入embed层进行一个dense网络，获得每个expert的权重
            exclusive_gates = []
            for i in range(exclusive_expert_num + common_expert_num):  # 有专家就有几个输出
                gate_out = mlp_layer(
                    name_scope="basic", inputs=embed_input_reshape,
                    units=[common_expert_num + exclusive_expert_num // target_num],
                    use_bn=use_bn, bn_decay=bn_decay, bn_epsilon=bn_epsilon,
                    use_drop=use_drop, drop_rate=drop_rate,
                    activation=[None], is_training=not is_test)  # [batch, expert_num]
                exclusive_gates.append(
                    tf.expand_dims(tf.nn.softmax(gate_out), axis=1))  # list([batch, 1, expert_num]*target_num)
            common_gates = []
            for i in range(exclusive_expert_num + common_expert_num):  # 有专家就有几个输出
                gate_out = mlp_layer(
                    name_scope="basic", inputs=embed_input_reshape,
                    units=[common_expert_num + exclusive_expert_num],
                    use_bn=use_bn, bn_decay=bn_decay, bn_epsilon=bn_epsilon,
                    use_drop=use_drop, drop_rate=drop_rate,
                    activation=[None], is_training=not is_test)  # [batch, expert_num]
                common_gates.append(
                    tf.expand_dims(tf.nn.softmax(gate_out), axis=1))  # list([batch, 1, expert_num]*target_num)

        # ----- build target -----
        with tf.variable_scope("Target-Weight-{}".format(i)):
            # 基于每个target进行加权
            # step = exclusive_expert_layers // target_num
            exclusive_target_input = []
            # gates[i] -> [batch,1,expert_num]; experts -> [batch,expert_num,export_out_size]
            for i in range(exclusive_expert_num):
                experts = exclusive_experts[i] + common_experts  # 拼接专属expert[i]和公共expert
                experts = tf.transpose(
                    tf.stack(experts, axis=0), perm=[1, 0, 2])
                exclusive_target_input.append(
                    tf.reshape(tf.matmul(exclusive_gates[i], experts),
                               shape=[-1, expert_out_size]))

            common_target_input = []
            for i in range(common_expert_num):
                experts = exclusive_experts + common_experts  # 拼接专属expert和公共expert
                experts = tf.transpose(
                    tf.stack(experts, axis=0), perm=[1, 0, 2])
                common_target_input.append(
                    tf.reshape(tf.matmul(common_gates[i], experts),
                               shape=[-1, expert_out_size]))

        exclusive_experts = exclusive_target_input
        common_experts = common_target_input

    # ----- build Gates -----
    with tf.variable_scope("Gate-Weight-end"):
        # 基于输入embed层进行一个dense网络，获得每个expert的权重
        gates = []
        for i in range(target_num):  # 有几个目标就有几个权重门
            gate_out = mlp_layer(
                name_scope="basic", inputs=embed_input_reshape,
                units=[common_expert_num + exclusive_expert_num // target_num],
                use_bn=use_bn, bn_decay=bn_decay, bn_epsilon=bn_epsilon,
                use_drop=use_drop, drop_rate=drop_rate,
                activation=[None], is_training=not is_test)  # [batch, expert_num]
            gates.append(tf.expand_dims(tf.nn.softmax(gate_out), axis=1))  # list([batch, 1, expert_num]*target_num)

    # ----- build target -----
    with tf.variable_scope("Target-Weight-end"):
        # 基于每个target进行加权
        step = exclusive_expert_num // target_num
        target_input = []
        for i in range(
                target_num):  # gates[i] -> [batch,1,expert_num]; experts -> [batch,expert_num,export_out_size]
            experts = exclusive_experts[i * step: (i + 1) * step] + common_experts  # 拼接专属expert和公共expert
            experts = tf.transpose(
                tf.stack(experts, axis=0), perm=[1, 0, 2])
            target_input.append(tf.reshape(tf.matmul(gates[i], experts), shape=[-1, expert_out_size]))
        # print(target_input)  -> target_input: list([batch, export_out_size]*target_num)

    # ---- build customize -----
    target_out = []  # 进入各自的tower里面
    for i in range(target_num):
        with tf.variable_scope("Target-Tower-{}".format(i)):
            y = mlp_layer(
                name_scope="basic", inputs=target_input[i], units=deep_layer_dim,
                use_bn=use_bn, bn_decay=bn_decay, bn_epsilon=bn_epsilon,
                use_drop=use_drop, drop_rate=drop_rate,
                activation=deep_act, is_training=not is_test)

            y_out_ = out_layer(name_scope="out", inputs=y, units=1)
            target_out.append(y_out_)

    with tf.variable_scope("ple_out"):  # 假设只有两个目标，对应ctr，cvr时
        pctr = target_out[0]
        pctcvr = target_out[0] * target_out[1]

    # =================================================
    # 统一模型output部分
    # =================================================
    pctr_ = tf.reshape(pctr, [-1])  # batch
    pctcvr_ = tf.reshape(pctcvr, [-1])  # batch
    loss1 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pctr_, labels=inputs["label"][:, 0]))

    loss2 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pctcvr_, labels=inputs["label"][:, 1]))

    loss = target1_wgt * loss1 + target2_wgt * loss2

    if is_test:
        tf.add_to_collections("input_tensor", embed_input)
        tf.add_to_collections("output_tensor", pctcvr_)
    net_dic = {
        "loss": loss,
        "ground_truth": inputs["label"][:, 1],
        "prediction": pctcvr_}
    # ==================================================
    return net_dic

# if __name__ == '__main__':
#     Invoke("conf/job.conf.dlbox.ini", mlp_model).run()
