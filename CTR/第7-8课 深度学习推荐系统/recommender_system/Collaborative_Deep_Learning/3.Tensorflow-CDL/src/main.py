from data_preprocessor import *

from CDL import CDL
from DAE import DAE
import tensorflow as tf
import time
import argparse

current_time = time.time()

parser = argparse.ArgumentParser(description='Collaborative Deep Learning')
parser.add_argument('--model_name', choices=['CDL'], default='CDL')
parser.add_argument('--data_name', choices=['politic_old','politic_new'], default='politic_new')
parser.add_argument('--test_fold', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--train_epoch', type=int, default=2)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3) # learning rate
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp','GradientDescent','Momentum'],default='Adam')
parser.add_argument('--keep_prob', type=float, default=0.9) # dropout
parser.add_argument('--a', type=float, default=1) # observed ratings
parser.add_argument('--b', type=float, default=0) # unseen ratings
parser.add_argument('--grad_clip', choices=['True', 'False'], default='True')  # True
parser.add_argument('--batch_normalization', choices=['True','False'], default = 'False')

parser.add_argument('--hidden_neuron', type=int, default=10)
parser.add_argument('--corruption_level', type=float, default=0.3) # input corruption ratio

parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Relu') # Encoder Activation
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Relu') # Decoder Activation

parser.add_argument('--encoder_method', choices=['SDAE'],default='SDAE')
parser.add_argument('--lambda_v',type=float , default = 1) # xi_dk prior std / cost3
parser.add_argument('--lambda_u',type=float , default = 0.1) # x_uk prior std / cost5
parser.add_argument('--lambda_w',type=float , default = 0.1) # SDAE weight std. / weight , bias regularization / cost1
parser.add_argument('--lambda_n',type=float , default = 1000) # SDAE output (cost2)
args = parser.parse_args()

random_seed = args.random_seed
tf.reset_default_graph()
np.random.seed(random_seed)
# np.random.RandomState
tf.set_random_seed(random_seed)

model_name = args.model_name
data_name = args.data_name
data_base_dir = "./data/"
path = data_base_dir + "%s" % data_name + "/"

if data_name == 'politic_new':
    num_users = 1537
    num_items = 7975
    num_total_ratings = 2999844
    num_voca = 13581
elif data_name == 'politic_old':
    num_users = 1540
    num_items = 7162
    num_total_ratings = 2779703
    num_voca = 10000
else:
    raise NotImplementedError("ERROR")

a = args.a
b = args.b

test_fold = args.test_fold
hidden_neuron = args.hidden_neuron

keep_prob = args.keep_prob
batch_normalization = args.batch_normalization

batch_size = 256
lr = args.lr
train_epoch = args.train_epoch
optimizer_method = args.optimizer_method
display_step = args.display_step
decay_epoch_step = 10000
decay_rate = 0.96
grad_clip = args.grad_clip

if args.f_act == "Sigmoid":
    f_act = tf.nn.sigmoid
elif args.f_act == "Relu":
    f_act = tf.nn.relu
elif args.f_act == "Tanh":
    f_act = tf.nn.tanh
elif args.f_act == "Identity":
    f_act = tf.identity
elif args.f_act == "Elu":
    f_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

if args.g_act == "Sigmoid":
    g_act = tf.nn.sigmoid
elif args.g_act == "Relu":
    g_act = tf.nn.relu
elif args.g_act == "Tanh":
    g_act = tf.nn.tanh
elif args.g_act == "Identity":
    g_act = tf.identity
elif args.g_act == "Elu":
    g_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

date = "0203"
result_path = './results/' + data_name + '/' + model_name + '/' + str(test_fold) +  '/' + str(current_time)+"/"

R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, data_name,num_users, num_items,num_total_ratings, a, b, test_fold,random_seed)

X_dw = read_bill_term(path,data_name,num_items,num_voca)

print ("Type of Model : %s" %model_name)
print ("Type of Data : %s" %data_name)
print ("# of User : %d" %num_users)
print ("# of Item : %d" %num_items)
print ("Test Fold : %d" %test_fold)
print ("Random seed : %d" %random_seed)
print ("Hidden neuron : %d" %hidden_neuron)


with tf.Session() as sess:
    if model_name == "CDL":
        lambda_u = args.lambda_u
        lambda_v = args.lambda_v
        lambda_w = args.lambda_w
        lambda_n = args.lambda_n
        lambda_list = [lambda_u, lambda_w, lambda_v, lambda_n]
        corruption_level = args.corruption_level

        layer_structure = [num_voca, 512, 128, hidden_neuron, 128, 512, num_voca]
        n_layer = len(layer_structure)
        pre_W = dict()
        pre_b = dict()

        for itr in range(n_layer - 1):
            initial_DAE = DAE(layer_structure[itr], layer_structure[itr + 1], num_items, num_voca, itr, "sigmoid")
            pre_W[itr], pre_b[itr] = initial_DAE.do_not_pretrain()
        model = CDL(sess, num_users, num_items, num_voca, hidden_neuron,current_time,
                  batch_size, lambda_list, layer_structure, train_epoch,
                  pre_W, pre_b, f_act,g_act,
                  corruption_level, keep_prob,
                  num_train_ratings, num_test_ratings,
                  X_dw, R, train_R, test_R, C,
                  mask_R, train_mask_R, test_mask_R,
                  grad_clip, display_step, a, b,
                  optimizer_method, lr,
                  result_path,
                  decay_rate, decay_epoch_step, args, random_seed, model_name, test_fold,data_name)
    model.run()


