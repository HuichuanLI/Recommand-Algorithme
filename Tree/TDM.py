import numpy as np
import time


### 定义树的节点
class TreeNode(object):
    def __init__(self, x, item_id=None):
        self.val = x
        self.item_id = item_id
        self.parent = None
        self.left = None
        self.right = None


### 这树是随机建树的基础类
class TreeInitialize(object):
    """build random binary tree"""

    def __init__(self, data):
        self.data = data[['item_ID', 'category_ID']]
        self.items = None
        self.root = None
        self.leaf_dict = {}
        self.node_size = 0

    ### 把item_id分配到category_id里
    def __random_sort(self):
        self.data = self.data.drop_duplicates()
        items_total = self.data.groupby(by=['category_ID']).apply(lambda x: x)
        self.items = items_total.tolist()
        return self.items

        ### 建树

    def _build_binary_tree(self, root, items):
        if len(items) == 1:
            leaf_node = TreeNode(0, item_id=items[0])
            leaf_node.parent = root.parent
            return leaf_node
        left_child, right_child = TreeNode(0), TreeNode(0)
        left_child.parent, right_child.parent = root, root
        mid = int(len(items) / 2)
        left = self._build_binary_tree(left_child, items[:mid])
        right = self._build_binary_tree(right_child, items[mid:])
        root.left = left
        root.right = right
        return root

        ### 给非叶子节点打上标号val,叶子节点就直接item_id表示了

    def _define_node_index(self, root):
        node_queue = [root]
        i = 0
        try:
            while node_queue:
                current_node = node_queue.pop(0)
                if current_node.left:
                    node_queue.append(current_node.left)
                if current_node.right:
                    node_queue.append(current_node.right)
                if current_node.item_id is not None:
                    self.leaf_dict[current_node.item_id] = current_node
                else:
                    current_node.val = i
                    i += 1
            self.node_size = i
            return 0
        except RuntimeError as err:
            print("Runtime Error Info: {}".format(err))
            return -1

    def random_binary_tree(self):
        root = TreeNode(0)
        items = self.__random_sort()
        self.root = self._build_binary_tree(root, items)
        _ = self._define_node_index(self.root)
        return self.root


class TreeLearning(TreeInitialize):
    """build the kmeans clustering tree."""

    def __init__(self, items, index_dict):
        self.items = items
        self.mapper = index_dict
        self.root = None
        self.leaf_dict = {}
        self.node_size = 0

    def _balance_clustering(self, c1, c2, item1, item2):
        amount = item1.shape[0] - item2.shape[0]
        if amount > 1:
            num = int(amount / 2)
            distance = np.sum(np.square(item1 - c1), axis=1)
            item_move = item1[distance.argsort()[-num:]]
            item2_adjust = np.concatenate((item2, item_move), axis=0)
            item1_adjust = np.delete(item1, distance.argsort()[-num:], axis=0)
        elif amount < -1:
            num = int(abs(amount) / 2)
            distance = np.sum(np.square(item - c2), axis=1)
            item_move = item2[distance.argsort()[-num:]]
            item1_adjust = np.concatenate((item1, item_move), axis=0)
            item2_adjust = np.delete(item2, distance.argsort()[-num:], axis=0)
        else:
            item1_adjust, item2_adjust = item1, item2
        return item1_adjust, item2_adjust

    def _k_means_clustering(self, items):
        m0, m1 = items[0], items[1]
        while True:
            indicate = np.sum(np.square(items - m1), axis=1) - np.sum(np.square(items - m2), axis=1)
            items_m1, items_m2 = items[indicate < 0], items[indicate > 0]
            m1_new = np.sum(items_m1, axis=0) / items_m1.shape[0]
            m2_new = np.sum(items_m2, axis=0) / items_m2.shape[0]
            if np.sum(np.absolute(m1_new - m1)) < 1e-3 and np.sum(np.absolute(m2_new - m2)) < 1e-3:
                break
            m1, m2 = m1_new, m2_new
        items_m1, items_m2 = self._balance_clutering(m1, m2, items_m1, items_m2)
        return items_m1, items_m2

    def _build_binary_tree(self, root, items):
        if items.shape[0] == 1:
            leaf_node = TreeNode(0, item_id=self.mapper[self.items.index(items[0].tolist())])
            leaf_node.parent = root.parent
            return leaf_node
        left_items, right_items = self._k_means_clustering(items)
        left_child, right_child = TreeNode(0), TreeNode(0)
        left_child.parent, right_child.parent = root, root
        left = self._build_binary_tree(left_child, left_items)
        right = self._build_binary_tree(right_child, right_items)
        root.left, root.right = left, right
        return root

    def clustering_binary_tree(self):
        root = TreeNode(0)
        items = np.array(self.items)
        self.root = self._build_binary_tree(root, items)
        _ = self._define_node_index(self.root)
        return self.root


import os
import time
import random
import multiprocessing as mp
import pandas as pd
import numpy as np


def data_process():
    """convert and split the raw data."""
    data_raw = pd.read_csv(LOAD_DIR, header=None,
                           names=['user_ID', 'item_ID', 'category_ID', 'behavior_type', 'timestamp'])
    data_raw = data_raw[:10000]
    user_list = data_raw.user_ID.drop_duplicates().to_list()
    user_dict = dict(zip(user_list, range(len(user_list))))
    data_raw['user_ID'] = data_raw.user_ID.apply(lambda x: user_dict[x])
    item_list = data_raw.item_ID.drop_duplicates().to_list()
    item_dict = dict(zip(item_list, range(len(item_list))))
    data_raw['item_ID'] = data_raw.item_ID.apply(lambda x: item_dict[x])
    category_list = data_raw.category_ID.drop_duplicates().to_list()
    category_dict = dict(zip(category_list, range(len(category_list))))
    data_raw['category_ID'] = data_raw.category_ID.apply(lambda x: category_dict[x])
    behavior_dict = dict(zip(['pv', 'buy', 'cart', 'fav'], range(4)))
    data_raw['behavior_type'] = data_raw.behavior_type.apply(lambda x: behavior_dict[x])
    time_window = _time_window_stamp()
    data_raw['timestamp'] = data_raw.timestamp.apply(_time_converter, args=(time_window,))
    random_tree = TreeInitialize(data_raw)
    _ = random_tree.random_binary_tree()
    data = data_raw.groupby(['user_ID', 'timestamp'])['item_ID'].apply(list).reset_index()
    data['behaviors'] = data_raw.groupby(['user_ID',
                                          'timestamp'])['behavior_type'].apply(list).reset_index()['behavior_type']
    data['behavior_num'] = data.behaviors.apply(lambda x: len(x))
    mask_length = data.behavior_num.max()
    data = data[data.behavior_num >= 10]
    data = data.drop(columns=['behavior_num'])
    data['item_ID'] = _mask_padding(data['item_ID'], mask_length)
    data['behaviors'] = _mask_padding(data['behaviors'], mask_length)
    data_train, data_validate, data_test = data[:-200], data[-200:-100], data[-100:]
    cache = (user_dict, item_dict, behavior_dict, random_tree)
    return data_train, data_validate.reset_index(drop=True), data_test.reset_index(drop=True), cache


def _time_window_stamp():
    boundaries = ['2017-11-26 00:00:00', '2017-11-27 00:00:00', '2017-11-28 00:00:00',
                  '2017-11-29 00:00:00', '2017-11-30 00:00:00', '2017-12-01 00:00:00',
                  '2017-12-02 00:00:00', '2017-12-03 00:00:00', '2017-12-04 00:00:00']
    for i in range(len(boundaries)):
        time_array = time.strptime(boundaries[i], "%Y-%m-%d %H:%M:%S")
        time_stamp = int(time.mktime(time_array))
        boundaries[i] = time_stamp
    return boundaries


def _time_converter(x, boundaries):
    tag = -1
    if x > boundaries[-1]:
        tag = 9
    else:
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
                tag = i
                break
    return tag


def _mask_padding(data, max_len):
    size = data.shape[0]
    raw = data.values
    mask = np.array([[-2] * max_len for _ in range(size)])
    for i in range(size):
        mask[i, :len(raw[i])] = raw[i]
    return mask.tolist()


class NeuralNet(object):
    """Deep network structure:
    input_embedding+node_embedding >>
    attention_block >>
    union_embedding >>
    MLP(128>64>24>2) >>
    label_probabilities.
    """

    def __init__(self, item_size, node_size, embedding_size):
        self.item_size = item_size
        self.embedding_size = embedding_size
        self.item_embeddings = tf.get_variable("item_embeddings",
                                               [self.item_size, self.embedding_size],
                                               use_resource=True)
        self.node_embeddings = tf.get_variable("node_embeddings",
                                               [node_size, self.embedding_size],
                                               use_resource=True)
        self.saver = None

    def _PRelu(self, x):
        m, n = tf.shape(x)
        value_init = 0.25 * tf.ones((1, n))
        a = tf.Variable(initial_value=value_init, use_resource=True)
        y = tf.maximum(x, 0) + a * tf.minimum(x, 0)
        return y

    def _activation_unit(self, item, node):
        item, node = tf.reshape(item, [1, -1]), tf.reshape(node, [1, -1])
        hybrid = item * node
        feature = tf.concat([item, hybrid, node], axis=1)
        layer1 = tf.layers.dense(feature, 36)
        layer1_prelu = self._PRelu(layer1)
        weight = tf.layers.dense(layer1_prelu, 1)
        return weight

    def _attention_feature(self, item, node, is_leafs, features):
        item_clip = item[item != -2]
        item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_clip)
        if is_leafs[0] == 0:
            node_embedding = tf.nn.embedding_lookup(self.node_embeddings, node)
        else:
            node_embedding = tf.nn.embedding_lookup(self.item_embeddings, node)
        item_num, _ = tf.shape(item_embedding)
        item_feature = None
        for i in range(item_num):
            item_weight = self._activation_unit(item_embedding[i], node_embedding[0])[0][0]
            if item_feature is None:
                item_feature = item_weight * item_embedding[i]
            else:
                item_feature = tf.add(item_feature, item_weight * item_embedding[i])
        item_feature = tf.concat([tf.reshape(item_feature, [1, -1]), node_embedding], axis=1)
        if features is None:
            features = item_feature
        else:
            features = tf.concat([features, item_feature], axis=0)
        return features

    def _attention_block(self, items, nodes, is_leafs):
        batch, _ = tf.shape(items)
        features = None
        for i in range(batch):
            features = self._attention_feature(items[i], nodes[i], is_leafs[i], features)
        return features

    def _network_structure(self, items, nodes, is_leafs, is_training):
        batch_features = self._attention_block(items, nodes, is_leafs)
        layer1 = tf.layers.dense(batch_features, 128)
        layer1_prelu = self._PRelu(layer1)
        layer1_bn = tf.layers.batch_normalization(layer1_prelu, training=is_training)
        layer2 = tf.layers.dense(layer1_bn, 64)
        layer2_prelu = self._PRelu(layer2)
        layer2_bn = tf.layers.batch_normalization(layer2_prelu, training=is_training)
        layer3 = tf.layers.dense(layer2_bn, 24)
        layer3_prelu = self._PRelu(layer3)
        layer3_bn = tf.layers.batch_normalization(layer3_prelu, training=is_training)
        logits = tf.layers.dense(layer3_bn, 2)
        return logits

    def _check_accuracy(self, iter_epoch, validate_data, is_training):
        num_correct, num_samples = 0, 0
        for items_val, nodes_val, is_leafs_val, labels_val in validate_data:
            scores = self._network_structure(items_val, nodes_val, is_leafs_val, is_training)
            scores = scores.numpy()
            label_predict = scores.argmax(axis=1)
            label_true = labels_val.argmax(axis=1)
            label_predict = label_predict[label_predict == label_true]
            label_predict = label_predict[label_predict == 0]
            label_true = label_true[label_true == 0]
            num_samples += label_true.shape[0]
            num_correct += label_predict.shape[0]
        accuracy = float(num_correct) / num_samples
        print("Iteration {}, total positive samples: {}, "
              "correct samples: {}, accuracy: {}".format(iter_epoch, num_samples, num_correct, accuracy))

    def train(self, use_gpu=False, train_data=None, validate_data=None,
              lr=0.001, b1=0.9, b2=0.999, eps=1e-08, num_epoch=10, check_epoch=200, save_epoch=1000):
        device = '/device:GPU:0' if use_gpu else '/cpu:0'
        with tf.device(device):
            container = tf.contrib.eager.EagerVariableStore()
            check_point = tf.contrib.eager.Checkpointable()
            iter_epoch = 0
            for epoch in range(num_epoch):
                print("Start epoch %d" % epoch)
                for items_tr, nodes_tr, is_leafs_tr, labels_tr in train_data:
                    with tf.GradientTape() as tape:
                        with container.as_default():
                            scores = self._network_structure(items_tr, nodes_tr, is_leafs_tr, 1)
                        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_tr, logits=scores)
                        loss = tf.reduce_sum(loss)
                        print("Epoch {}, Iteration {}, loss {}".format(epoch, iter_epoch, loss))
                    gradients = tape.gradient(loss, container.trainable_variables())
                    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps)
                    optimizer.apply_gradients(zip(gradients, container.trainable_variables()))
                    if iter_epoch % check_epoch == 0:
                        self._check_accuracy(iter_epoch, validate_data, 0)
                    if iter_epoch % save_epoch == 0:
                        for k, v in container._store._vars.items():
                            setattr(check_point, k, v)
                        self.saver = tf.train.Checkpoint(checkpointable=check_point)
                        self.saver.save(MODEL_NAME)
                    iter_epoch += 1
        print("It's completed to train the network.")

    def get_embeddings(self, item_list, use_gpu=True):
        """
        TODO: validate and optimize
        """
        model_path = tf.train.latest_checkpoint(MODEL_DIR + '/models/')
        self.saver.restore(model_path)
        device = '/device:GPU:0' if use_gpu else '/cpu:0'
        with tf.device(device):
            item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, np.array(item_list))
            res = item_embeddings.numpy()
        return res.tolist()

    def predict(self, data, use_gpu=True):
        """
        TODO: validate and optimize
        """
        model_path = tf.train.latest_checkpoint(MODEL_DIR + '/models/')
        self.saver.restore(model_path)
        device = '/device:GPU:0' if use_gpu else '/cpu:0'
        with tf.device(device):
            items, nodes, is_leafs = data
            scores = self._network_structure(items, nodes, is_leafs, 0)
            scores = scores.numpy()
        return scores[:, 0]
