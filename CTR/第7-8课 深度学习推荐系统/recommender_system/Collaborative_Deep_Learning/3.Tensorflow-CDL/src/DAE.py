import tensorflow as tf
import numpy as np
import time

class DAE():
    def __init__(self,n_visible,n_hidden,num_item,num_voca,itr,dipen_activation,lambda_w=33.33333):
        self.num_item = num_item
        self.num_voca = num_voca
        self.itr = itr
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.Laeyrs = []
        self.display_step = 50
        self.dipen_activation = dipen_activation
        self.lambda_w = lambda_w

    def do_not_pretrain(self):
        with tf.variable_scope("SDAE_Variable"):
            pre_W = tf.get_variable(name=("pre_W"+str(self.itr)), initializer=tf.truncated_normal(shape=[self.n_visible, self.n_hidden],
                                                                      mean=0, stddev=tf.truediv(1.0,self.lambda_w)), dtype=tf.float32)
            pre_b = tf.get_variable(name=("pre_b"+str(self.itr)), initializer=tf.zeros(shape=self.n_hidden), dtype=tf.float32)
            '''
            pre_W = tf.get_variable(name=("pre_W"+str(self.itr)), shape=[self.n_visible, self.n_hidden], dtype=tf.float32,initializer=tf.random_normal_initializer())
            pre_b = tf.get_variable(name=("pre_b"+str(self.itr)), shape=[self.n_hidden], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
         '''
        return pre_W , pre_b

    def do_pretrain(self,pretrain_input,epoch,batch_size,learning_rate,dropout,corruption_level):
        current_time = str(time.time())
        '''
        fan_in = self.n_visible
        fan_out = self.n_hidden
        if self.dipen_activation == "sigmoid":
            tmp_const = 4
        elif self.dipen_activation == "tanh":
            tmp_const = 1
        elif self.dipen_activation == "relu6":
            tmp_const = 1
        #low = -1 * tmp_const * np.sqrt(6.0 / float(fan_in + fan_out))  # use 4 for sigmoid, 1 for tanh activation
        #high = 1 * tmp_const * np.sqrt(6.0 / float(fan_in + fan_out))
        '''
        pre_W1 = tf.get_variable(name=("pre_W1"+str(self.itr)), shape=[self.n_visible, self.n_hidden], dtype=tf.float32,initializer=tf.random_normal_initializer())
        pre_b1 = tf.get_variable(name=("pre_b1"+str(self.itr)), shape=[self.n_hidden], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())

        pre_W2 = tf.get_variable(name=("pre_W2"+str(self.itr)), shape=[self.n_hidden, self.n_visible], dtype=tf.float32,initializer=tf.random_normal_initializer())
        pre_b2 = tf.get_variable(name=("pre_b2"+str(self.itr)), shape=[self.n_visible], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())

        X = tf.placeholder("float", [None, self.n_visible])
        mask = tf.placeholder("float", [None,self.n_visible],name='mask')
        mask_np = np.random.binomial(1, 1 - corruption_level, (self.num_item, self.n_visible))
        corrupted_X = tf.mul(mask , X)

        # encoder
        if self.dipen_activation == "sigmoid":
            hidden = tf.nn.sigmoid(tf.add(tf.matmul(corrupted_X, pre_W1), pre_b1))
        elif self.dipen_activation == "tanh":
            hidden = tf.nn.tanh(tf.add(tf.matmul(corrupted_X, pre_W1), pre_b1))
        elif self.dipen_activation == "relu6":
            hidden = tf.nn.relu6(tf.add(tf.matmul(corrupted_X, pre_W1), pre_b1))
        elif self.dipen_activation == "relu":
            hidden = tf.nn.relu(tf.add(tf.matmul(corrupted_X, pre_W1), pre_b1))
        elif self.dipen_activation == "elu":
            hidden = tf.nn.elu(tf.add(tf.matmul(corrupted_X, pre_W1), pre_b1))

        keep_prob = tf.placeholder(tf.float32)
        hidden = tf.nn.dropout(hidden,dropout) # probability to keep units

        if self.dipen_activation == "sigmoid":
            output = tf.nn.sigmoid(tf.add(tf.matmul(hidden, pre_W2), pre_b2))
        elif self.dipen_activation == "tanh":
            output = tf.nn.tanh(tf.add(tf.matmul(hidden, pre_W2), pre_b2))
        elif self.dipen_activation == "relu6":
            output = tf.nn.relu6(tf.add(tf.matmul(hidden, pre_W2), pre_b2))
        elif self.dipen_activation == "relu":
            output = tf.nn.relu(tf.add(tf.matmul(hidden, pre_W2), pre_b2))

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(output - X, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-1).minimize(cost)
        #optimizer = tf.train.AdadeltaOptimizer.__init__(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
        # Initializing the variables
        init = tf.initialize_all_variables()
        with tf.Session() as sess:

            sess.run(init)
            total_batch = int(self.num_item / batch_size)
            random_perm_doc_idx = list(np.random.permutation(self.num_item))
            # Training cycle
            for epoch in range(epoch):
                cost_list = []
                # Loop over all batches
                for i in range(total_batch):
                    if i == total_batch - 1:
                        batch_set_idx = random_perm_doc_idx[i * batch_size:]
                    elif i < total_batch - 1:
                        batch_set_idx = random_perm_doc_idx[i * batch_size: (i + 1) * batch_size]

                    batch_xs = pretrain_input[batch_set_idx, :]
                    batch_mask_np = mask_np[batch_set_idx, :]
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs , keep_prob:dropout , mask:batch_mask_np})
                    cost_list.append(c)
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    final_cost = np.sum(cost_list)
                    print("Training (%d)-th DAE" %self.itr, "Epoch : %d" %(epoch + 1), "cost = {:.20f}".format(final_cost))
            pretrain_input = tf.convert_to_tensor(pretrain_input)
            pretrain_input = tf.cast(pretrain_input,tf.float32)
            next_pretrain_input = tf.nn.sigmoid(tf.add(tf.matmul(pretrain_input, pre_W1), pre_b1))  # encoder
            next_pretrain_input = next_pretrain_input.eval()
        return pre_W1 , pre_b1 , next_pretrain_input