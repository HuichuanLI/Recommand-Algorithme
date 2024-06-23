from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import xDeepFM, DCN, DIN, DIEN, DIFM, MMOE, ESMM, PLE


def t0():
    sparse_features = ['c1', 'c2', 'c3']  # 模拟存在三个离散特征属性
    dense_features = ['d1', 'd2']  # 模拟存在两个连续特征属性
    sparse_feature_columns = []
    dense_feature_columns = []
    # 离散特征元数据构造
    for i, feat in enumerate(sparse_features):
        sparse_feature_columns.append(SparseFeat(feat, vocabulary_size=5 + i, embedding_dim=4))
    # 连续特征元数据构造
    for feat in dense_features:
        dense_feature_columns.append(DenseFeat(feat, 1, ))
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    linear_feature_columns = sparse_feature_columns + dense_feature_columns

    model = DCN(linear_feature_columns, dnn_feature_columns)

    print(model)


def t1():
    # https://blog.csdn.net/weixin_42357472/article/details/115250346
    sparse_features = ['c1', 'c2', 'c3', 'hist_spu_ids', 'hist_shop_ids']  # 模拟存在三个普通离散特征属性 + 2个序列特征
    dense_features = ['d1', 'd2']  # 模拟存在两个连续特征属性
    sparse_feature_columns = []
    dense_feature_columns = []
    # 离散特征元数据构造
    for i, feat in enumerate(sparse_features):
        if feat.startswith("hist_"):
            sparse_feat = SparseFeat(feat[5:], vocabulary_size=5 + i, embedding_dim=4)
            sparse_feature_columns.append(sparse_feat)

            sparse_feat = VarLenSparseFeat(
                SparseFeat(feat, vocabulary_size=5 + i, embedding_dim=4),
                maxlen=50, length_name='length_name'
            )
            sparse_feature_columns.append(sparse_feat)
        else:
            sparse_feat = SparseFeat(feat, vocabulary_size=5 + i, embedding_dim=4)
            sparse_feature_columns.append(sparse_feat)
    # 连续特征元数据构造
    for feat in dense_features:
        dense_feature_columns.append(DenseFeat(feat, 1, ))
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns

    model = DIN(dnn_feature_columns, history_feature_list=['spu_ids', 'shop_ids'])

    print(model)


def t2():
    import numpy as np
    import torch

    def get_xy_fd():
        feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
                           SparseFeat('item', 3 + 1, embedding_dim=8),
                           SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                           DenseFeat('score', 1)]

        feature_columns += [
            VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), maxlen=4,
                             length_name="seq_length")]
        behavior_feature_list = ["item", "item_gender"]
        uid = np.array([0, 1, 2])
        ugender = np.array([0, 1, 0])
        iid = np.array([1, 2, 3])  # 0 is mask value
        igender = np.array([1, 2, 1])  # 0 is mask value
        score = np.array([0.1, 0.2, 0.3])

        hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
        behavior_length = np.array([3, 3, 2])

        feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                        'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
                        "seq_length": behavior_length}
        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
        y = np.array([1, 0, 1])

        return x, y, feature_columns, behavior_feature_list

    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    print(model)
    model.compile('adagrad', 'binary_crossentropy', metrics=['binary_crossentropy'])
    history = model.fit(x, y, batch_size=3, epochs=10, verbose=2, validation_split=0.0)
    print(history)


def t3():
    dnn_feature_columns = [
        SparseFeat('user', 3, embedding_dim=8),
        SparseFeat('gender', 2, embedding_dim=8),
        SparseFeat('item', 3 + 1, embedding_dim=8),
        SparseFeat('item_gender', 2 + 1, embedding_dim=8),
        DenseFeat('score', 1)
    ]
    model = MMOE(dnn_feature_columns, num_experts=3, task_names=('a', 'b', 'c'),
                 task_types=('binary', 'binary', 'binary'))
    print(model)


def t4():
    dnn_feature_columns = [
        SparseFeat('user', 3, embedding_dim=8),
        SparseFeat('gender', 2, embedding_dim=8),
        SparseFeat('item', 3 + 1, embedding_dim=8),
        SparseFeat('item_gender', 2 + 1, embedding_dim=8),
        DenseFeat('score', 1)
    ]
    model = ESMM(dnn_feature_columns)
    print(model)


def t5():
    # https://deepctr-doc.readthedocs.io/en/v0.9.1/Examples.html#multitask-learning-mmoe
    dnn_feature_columns = [
        SparseFeat('user', 3, embedding_dim=8),
        SparseFeat('gender', 2, embedding_dim=8),
        SparseFeat('item', 3 + 1, embedding_dim=8),
        SparseFeat('item_gender', 2 + 1, embedding_dim=8),
        DenseFeat('score', 1)
    ]
    model = PLE(dnn_feature_columns)
    print(model)


if __name__ == '__main__':
    t5()
