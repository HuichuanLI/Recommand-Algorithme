# -*- coding:utf-8 -*-
# @Time : 2021/8/4 11:00 下午
# @Author : huichuan LI
# @File : DSSM.py
# @Software: PyCharm

import keras.backend as k


def swish(x):
    return K.


def get_DSSMModel(vocab_size_user, vocab_size_item, embed_size_user=64, embed_size_item=64, activation='swish',
                  item_embedding_weight=None, user_embedding_weight=None):
    input_u = Input((1,), name='u_input')

    input_x = Input((2,), name='x_input')

    input_xp = Lambda(lambda x: x[:, 0], name='lambda_p')(input_x)
    input_xn = Lambda(lambda x: x[:, 1], name='lambda_n')(input_x)

    input_xp = Reshape((1,), name='reshape_p')(input_xp)

    input_xn = Reshape((1,), name='reshape_n')(input_xn)
    base_model = get_two_stream_model(vocab_size_user=vocab_size_user,
                                      vocab_size_item=vocab_size_item,
                                      embed_size_user=embed_size_user,
                                      embed_size_item=embed_size_item,
                                      activation=activation,
                                      block_name='BaseModel',
                                      item_embedding_weight=item_embedding_weight,
                                      user_embedding_weight=user_embedding_weight)
    score_p = base_model([input_u, input_xp])
    score_n = base_model([input_u, input_xn])
    delta_score = Lambda(lambda x: x[0] - x[1], name='delta_score')([score_p, score_n])
    model = Model(inputs=[input_u, input_x], outputs=[delta_scorel], name='DSSM_Model')
    model.save = base_model.save

    if __name__ == "__main__":
        model = get_DSSMModel(100, 10, embed_size_user=64, embed_size_item=64, activation='swish',
                              item_embedding_weight=None, user_embedding_weight=None)
