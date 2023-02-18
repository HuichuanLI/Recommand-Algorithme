import tensorflow as tf

weight_dim = 17  # 16 + 1
learning_rate = 0.01
feature_dim = 4


def fm_fn(inputs, is_test):
    weight = tf.reshape(inputs["feature_embedding"],
                        shape=[-1, feature_dim, weight_dim])

    # batch * 4 * 16, batch * 4 * 1
    cross_weight, linear_weight = tf.split(
        weight, num_or_size_splits=[weight_dim - 1, 1], axis=2
    )

    bias = tf.get_variable("bias", [1, ], initializer=tf.zeros_initializer())

    linear_model = tf.nn.bias_add(tf.reduce_sum(linear_weight, axis=1), bias)

    square_sum = tf.square(tf.reduce_sum(cross_weight, axis=1))  #
    summed_square = tf.reduce_sum(tf.square(cross_weight), axis=1)

    cross_model = 0.5 * tf.reduce_sum(tf.subtract(square_sum, summed_square),
                                      axis=1, keep_dims=True)
    y_pred = cross_model + linear_model

    y_sigmoid = tf.sigmoid(y_pred)

    if is_test:
        tf.compat.v1.add_to_collections("input_tensor", weight)
        tf.compat.v1.add_to_collections("output_tensor", y_sigmoid)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=inputs[
                                                                      "label"]))

    model_result = {
        "loss": loss,
        "label": inputs["label"][:, 0],
        "prediction": y_pred[:, 0]
    }
    return model_result


# @tf.function
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.compat.v1.variable_scope("net_graph", reuse=is_test):
        model_result_dict = fm_fn(inputs, is_test)
        result["result"] = model_result_dict
        if is_test:
            return result
        loss = model_result_dict["loss"]
        embedding_gradient = tf.gradients(loss, [inputs["feature_embedding"]],
                                          name="feature_embedding")[0]
        result["new_feature_embedding"] = inputs[
                                              "feature_embedding"] - learning_rate * embedding_gradient
        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]

        return result
