import numpy as np
import os
from numpy import inf
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm
import functools

def evaluation(test_R,test_mask_R,Estimated_R,num_test_ratings):

    pre_numerator = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.square(pre_numerator))
    denominator = num_test_ratings
    RMSE = np.sqrt(numerator / float(denominator))

    pre_numeartor = np.multiply((test_R - Estimated_R), test_mask_R)
    numerator = np.sum(np.abs(pre_numeartor))
    denominator = num_test_ratings
    MAE = numerator / float(denominator)

    pre_numeartor1 = np.sign(Estimated_R - 0.5)
    tmp_test_R = np.sign(test_R - 0.5)

    pre_numerator2 = np.multiply((pre_numeartor1 == tmp_test_R), test_mask_R)
    numerator = np.sum(pre_numerator2)
    denominator = num_test_ratings
    ACC = numerator / float(denominator)

    a = np.log(Estimated_R)
    b = np.log(1 - Estimated_R)
    a[a == -inf] = 0
    b[b == -inf] = 0

    tmp_r = test_R
    tmp_r = a * (tmp_r > 0) + b * (tmp_r == 0)
    tmp_r = np.multiply(tmp_r, test_mask_R)
    numerator = np.sum(tmp_r)
    denominator = num_test_ratings
    AVG_loglikelihood = numerator / float(denominator)

    return RMSE,MAE,ACC,AVG_loglikelihood

def make_records(result_path,test_acc_list,test_rmse_list,test_mae_list,test_avg_loglike_list,current_time,
                 args,model_name,data_name,train_ratio,hidden_neuron,random_seed,optimizer_method,lr):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    overview = './results/' + 'overview.txt'
    basic_info = result_path + "basic_info.txt"
    test_record = result_path + "test_record.txt"

    with open(test_record, 'w') as g:

        g.write(str("ACC:"))
        g.write('\t')
        for itr in range(len(test_acc_list)):
            g.write(str(test_acc_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RMSE:"))
        g.write('\t')
        for itr in range(len(test_rmse_list)):
            g.write(str(test_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("MAE:"))
        g.write('\t')
        for itr in range(len(test_mae_list)):
            g.write(str(test_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("AVG Likelihood:"))
        g.write('\t')
        for itr in range(len(test_avg_loglike_list)):
            g.write(str(test_avg_loglike_list[itr]))
            g.write('\t')
        g.write('\n')

    with open(basic_info, 'w') as h:
        h.write(str(args))

    with open(overview, 'a') as f:
        f.write(str(data_name))
        f.write('\t')
        f.write(str(model_name))
        f.write('\t')
        f.write(str(train_ratio))
        f.write('\t')
        f.write(str(current_time))
        f.write('\t')
        f.write(str(test_rmse_list[-1]))
        f.write('\t')
        f.write(str(test_mae_list[-1]))
        f.write('\t')
        f.write(str(test_acc_list[-1]))
        f.write('\t')
        f.write(str(test_avg_loglike_list[-1]))
        f.write('\t')
        f.write(str(hidden_neuron))
        f.write('\t')
        f.write(str(args.corruption_level))
        f.write('\t')

        f.write(str(args.lambda_u))
        f.write('\t')
        f.write(str(args.lambda_w))
        f.write('\t')
        f.write(str(args.lambda_n))
        f.write('\t')
        f.write(str(args.lambda_v))
        f.write('\t')
        f.write(str(args.f_act))
        f.write('\t')
        f.write(str(args.g_act))
        f.write('\n')

    Test = plt.plot(test_acc_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(result_path + "ACC.png")
    plt.clf()

    Test = plt.plot(test_rmse_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(result_path + "RMSE.png")
    plt.clf()

    Test = plt.plot(test_mae_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(result_path + "MAE.png")
    plt.clf()

    Test = plt.plot(test_avg_loglike_list, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Test AVG likelihood')
    plt.legend()
    plt.savefig(result_path + "AVG.png")
    plt.clf()

def variable_save(result_path,model_name,train_var_list1,train_var_list2,Estimated_R,test_v_ud,mask_test_v_ud):
    for var in train_var_list1:
        var_value = var.eval()
        var_name = ((var.name).split('/'))[1]
        var_name = (var_name.split(':'))[0]
        np.savetxt(result_path + var_name , var_value)

    for var in train_var_list2:
        if model_name == "DIPEN_with_VAE":
            var_value = var.eval()
            var_name = (var.name.split(':'))[0]
            print (var_name)
            var_name = var_name.replace("/","_")
            #var_name = ((var.name).split('/'))[2]
            #var_name = (var_name.split(':'))[0]
            print (var.name)
            print (var_name)
            print ("================================")
            np.savetxt(result_path + var_name, var_value)
        else:
            var_value = var.eval()
            var_name = ((var.name).split('/'))[1]
            var_name = (var_name.split(':'))[0]
            np.savetxt(result_path + var_name , var_value)

    Estimated_R = np.where(Estimated_R<0.5,0,1)
    Error_list = np.nonzero( (Estimated_R - test_v_ud) * mask_test_v_ud )
    user_error_list = Error_list[0]
    item_error_list = Error_list[1]
    np.savetxt(result_path+"Estimated_R",Estimated_R)
    np.savetxt(result_path+"test_v_ud",test_v_ud)
    np.savetxt(result_path+"mask_test_v_ud",mask_test_v_ud)
    np.savetxt(result_path + "user_error_list", user_error_list)
    np.savetxt(result_path + "item_error_list", item_error_list)

def SDAE_calculate(model_name,X_c, layer_structure, W, b, batch_normalization, f_act,g_act, model_keep_prob,V_u=None):
    hidden_value = X_c
    for itr1 in range(len(layer_structure) - 1):
        ''' Encoder '''
        if itr1 <= int(len(layer_structure) / 2) - 1:
            if (itr1 == 0) and (model_name == "CDAE"):
                ''' V_u '''
                before_activation = tf.add(tf.add(tf.matmul(hidden_value, W[itr1]),V_u), b[itr1])
            else:
                before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = f_act(before_activation)
            ''' Decoder '''
        elif itr1 > int(len(layer_structure) / 2) - 1:
            before_activation = tf.add(tf.matmul(hidden_value, W[itr1]), b[itr1])
            if batch_normalization == "True":
                before_activation = batch_norm(before_activation)
            hidden_value = g_act(before_activation)
        if itr1 < len(layer_structure) - 2: # add dropout except final layer
            hidden_value = tf.nn.dropout(hidden_value, model_keep_prob)
        if itr1 == int(len(layer_structure) / 2) - 1:
            Encoded_X = hidden_value

    sdae_output = hidden_value

    return Encoded_X, sdae_output

def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

def softmax(w, t = 1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist