"""
@author: Xiao Ning
Benchmark Code of SIRD model
2022-07-12
"""
import os
from sre_parse import FLAGS
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from datetime import datetime
import platform
import shutil
import modelUtils
import DNN_tools
import DNN_data
import plotData
import saveData
import argparse
import dataUtils

# tf2兼容tf1
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('--data_fname', type=str, default='./data/minnesota3.csv',
                    help='data path')

parser.add_argument('--output_dir', type=str, default='./output_SIRD',
                    help='data path')

parser.add_argument('--train_epoches', type=int, default=100000,
                    help='max epoch during training.')

parser.add_argument('--eval_epoches', type=int, default=2000,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer')

parser.add_argument('--batch_size', type=int, default=16,
                    help='.')

parser.add_argument('--sample_method', type=str, default='rand_sample_sort',
                    help='. : random_sample, rand_sample_sort, sequential_sort')

parser.add_argument('--sird_network', type=str, default='DNN_FOURIERBASE',
                    help='network archtecture:' 'DNN, DNN_FOURIERBASE, DNN_SCALE')

parser.add_argument('--params_network', type=str, default='DNN_FOURIERBASE',
                    help='network archtecture:' 'DNN, DNN_FOURIERBASE, DNN_SCALE')

parser.add_argument('--hidden_sird', type=set, default=([35, 50, 30, 30, 20]),
                    help='hidden layers:'
                         '(80, 80, 60, 40, 40, 20)'
                         '(100, 100, 80, 60, 60, 40)'
                         '(200, 100, 100, 80, 50, 50)')

parser.add_argument('--hidden_params', type=set, default=([35, 50, 30, 30, 20]),
                    help='hidden layers:'
                         '(80, 80, 60, 40, 40, 20)'
                         '(100, 100, 80, 60, 60, 40)'
                         '(200, 100, 100, 80, 50, 50)')  

parser.add_argument('--loss_function', type=str, default='L2_loss',
                    help='loss function:' 'L2_loss, lncosh_loss')      
# SIRD和参数网络模型的激活函数的选择
parser.add_argument('--activateIn_sird', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--activate_sird', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--activateIn_params', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--activate_params', type=str, default='tanh',
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')                                     

parser.add_argument('--init_penalty2predict_true', type=int, default=50, # 预测值和真值的误差惩罚因子初值,用于处理具有真实值的变量
                    help='Regularization parameter for boundary conditions.')

parser.add_argument('--activate_stage_penalty', type=bool, default=True,
                    help='Whether to use Regularization parameter for boundary conditions.')

parser.add_argument('--regular_method', type=str, default='L2',
                    help='The method of regular weights and biases:' 'L0, L1')

parser.add_argument('--regular_weight', type=float, default=0.00005, # 神经网络参数的惩罚因子
                    help='Regularization parameter for weights.' '0.00001, 0.00005, 0.0001, 0.0005, 0.001')

parser.add_argument('--initial_learning_rate', type=float, default=0.002,
                    help='.'
                    '0.1, 0.01, 0.05, 0,001')
parser.add_argument('--decay_steps', type=float, default=1000,
                    help='.' '0.1, 0.01, 0.05, 0,001')
parser.add_argument('--decay_rate', type=float, default=0.90,
                    help='.' '0.1, 0.01, 0.05, 0,001')

parser.add_argument('--population', type=float, default=3450000,
                    help='.')

parser.add_argument('--input_dim', type=float, default=1,
                    help='.')

parser.add_argument('--output_dim', type=float, default=1,
                    help='.')


def act_gauss(input):
    # out = tf.exp(-0.25*tf.multiply(input, input))
    out = tf.exp(-0.5 * tf.multiply(input, input))
    return out

# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of dealing with parameters: %s\n' % str(R_dic['model2paras']), log_fileout)

    if str.upper(R_dic['model2paras']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for parameter: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for parameter: %s\n' % str(R_dic['actIn_Name2paras']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for parameter: %s\n' % str(R_dic['act_Name2paras']), log_fileout)

    DNN_tools.log_string('hidden layers for parameters: %s\n' % str(R_dic['hidden2para']), log_fileout)

    if str.upper(R_dic['model2paras']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2paras']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for para-NN: %s\n' % str(R_dic['if_repeat_High_freq2paras']), log_fileout)

    DNN_tools.log_string('The training model for all networks: %s\n' % str(R_dic['train_model']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                             log_fileout)
    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    # DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    # DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    # DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(epoch, run_time, tmp_lr, penalty_wb2beta, penalty_wb2gamma, penalty_wb2mu, loss_i,
                        loss_r, loss_d, loss_all, log_out=None):
    print('train epoch: %d, time: %.3f' % (epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('penalty weights and biases for Beta: %.16f' % penalty_wb2beta)
    print('penalty weights and biases for Gamma: %.16f' % penalty_wb2gamma)
    print('penalty weights and biases for Mu: %.16f' % penalty_wb2mu)
    # print('loss for S: %.16f' % loss_s)
    print('loss for I: %.16f' % loss_i)
    print('loss for R: %.16f' % loss_r)
    print('loss for D: %.16f' % loss_d)
    print('total loss: %.16f\n' % loss_all)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %.10f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty weights and biases for Beta: %.16f' % penalty_wb2beta, log_out)
    DNN_tools.log_string('penalty weights and biases for Gamma: %.16f' % penalty_wb2gamma, log_out)
    DNN_tools.log_string('penalty weights and biases for Mu: %.16f' % penalty_wb2mu, log_out)
    # DNN_tools.log_string('loss for S: %.16f' % loss_s, log_out)
    DNN_tools.log_string('loss for I: %.16f' % loss_i, log_out)
    DNN_tools.log_string('loss for params: %.16f' % loss_r, log_out)
    DNN_tools.log_string('loss for D: %.16f' % loss_d, log_out)
    DNN_tools.log_string('total loss: %.16f \n\n' % loss_all, log_out)


# Reference paper: A flexible rolling regression framework for the identification of time-varying SIRD models
def solve_SIRD2COVID(params):
    log_out_path = params['FolderName']        # 将路径从字典 params 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(params, log_fileout)

    # trainSet_szie = params['size2train']                   # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchSize_train = FLAGS.batch_size           # 训练批量的大小,该值远小于训练集大小
    # batchSize_test = params['batch_size2test']             # 测试批量的大小,该值小于等于测试集大小
    wb_penalty = params['regular_weight']                  # 神经网络参数的惩罚因子
    lr_decay = params['lr_decay']                          # 学习率额衰减
    init_lr = params['learning_rate']                      # 初始学习率

    act_func2paras = params['act_Name2paras']              # 参数网络的隐藏层激活函数

    input_dim = FLAGS.input_dim
    out_dim = FLAGS.output_dim
    hidden_para = params['hidden2para']

    AI = tf.eye(batchSize_train, dtype=tf.float32) * (-2)
    Ones_mat = tf.ones([batchSize_train, batchSize_train], dtype=tf.float32)
    A_diag = tf.linalg.band_part(Ones_mat, 0, 1)
    Amat = AI + A_diag
    Amat

    if params['model2paras'].upper() == 'DNN_FOURIERBASE':
        Weight2beta, Bias2beta = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, 'wb_beta')
        Weight2gamma, Bias2gamma = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, 'wb_gamma')
        Weight2mu, Bias2mu = modelUtils.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, 'wb_mu')
    else:
        Weight2beta, Bias2beta = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_para, 'wb_beta')
        Weight2gamma, Bias2gamma = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_para, 'wb_gamma')
        Weight2mu, Bias2mu = modelUtils.Xavier_init_NN(input_dim, out_dim, hidden_para, 'wb_mu')

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (params['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            T_train = tf.compat.v1.placeholder(tf.float32, name='T_train', shape=[None, out_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, out_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[None, out_dim])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[None, out_dim])

            # in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if 'DNN' == str.upper(params['model2paras']):
                in_beta2train = modelUtils.DNN(T_train, Weight2beta, Bias2beta, hidden_para,
                                             activateIn_name=params['actIn_Name2paras'], activate_name=params['act_Name2paras'])
                in_gamma2train = modelUtils.DNN(T_train, Weight2gamma, Bias2gamma, hidden_para,
                                              activateIn_name=params['actIn_Name2paras'], activate_name=params['act_Name2paras'])
                in_mu2train = modelUtils.DNN(T_train, Weight2mu, Bias2mu, hidden_para,
                                           activateIn_name=params['actIn_Name2paras'], activate_name=params['act_Name2paras'])
            elif 'DNN_SCALE' == str.upper(params['model2paras']):
                freq2paras = params['freq2paras']
                in_beta2train = modelUtils.DNN_scale(T_train, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activateIn_name=params['actIn_Name2paras'],
                                                   activate_name=params['act_Name2paras'])
                in_gamma2train = modelUtils.DNN_scale(T_train, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activateIn_name=params['actIn_Name2paras'],
                                                    activate_name=params['act_Name2paras'])
                in_mu2train = modelUtils.DNN_scale(T_train, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                 activateIn_name=params['actIn_Name2paras'],
                                                 activate_name=params['act_Name2paras'])
 
            elif str.upper(params['model2paras']) == 'DNN_FOURIERBASE':
                freq2paras = params['freq2paras']
                in_beta2train = modelUtils.DNN_FourierBase(T_train, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                         activate_name=params['act_Name2paras'], sFourier=1.0)
                in_gamma2train = modelUtils.DNN_FourierBase(T_train, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                          activate_name=params['act_Name2paras'], sFourier=1.0)
                in_mu2train = modelUtils.DNN_FourierBase(T_train, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                       activate_name=params['act_Name2paras'], sFourier=1.0)

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            betaNN2train = tf.nn.sigmoid(in_beta2train)
            gammaNN2train = tf.nn.sigmoid(in_gamma2train)
            # # muNN2train = 0.01*tf.nn.sigmoid(in_mu2train)
            # muNN2train = 0.05 * tf.nn.sigmoid(in_mu2train)
            muNN2train = 0.01 * tf.nn.sigmoid(in_mu2train)
            #
            # betaNN2train_test = tf.nn.sigmoid(in_beta2train_test)
            # gammaNN2train_test = tf.nn.sigmoid(in_gamma2train_test)
            # # # muNN2train_test = 0.01 * tf.nn.sigmoid(in_mu2train_test)
            # # muNN2train_test = 0.05 * tf.nn.sigmoid(in_mu2train_test)
            # muNN2train_test = 0.1 * tf.nn.sigmoid(in_mu2train_test)
            # #
            # betaNN2test = tf.nn.sigmoid(in_beta2test)
            # gammaNN2test = tf.nn.sigmoid(in_gamma2test)
            # # # muNN2test = 0.01*tf.nn.sigmoid(in_mu2test)
            # # muNN2test = 0.05 * tf.nn.sigmoid(in_mu2test)
            # muNN2test = 0.1 * tf.nn.sigmoid(in_mu2test)

            # betaNN2train = act_gauss(in_beta2train)
            # gammaNN2train = act_gauss(in_gamma2train)
            # # muNN2train = 0.01*act_gauss(in_mu2train)
            # muNN2train = 0.05 * act_gauss(in_mu2train)
            #
            # betaNN2train_test = act_gauss(in_beta2train_test)
            # gammaNN2train_test = act_gauss(in_gamma2train_test)
            # # muNN2train_test = 0.01 * act_gauss(in_mu2train_test)
            # muNN2train_test = 0.05 * act_gauss(in_mu2train_test)
            #
            # betaNN2test = act_gauss(in_beta2test)
            # gammaNN2test = act_gauss(in_gamma2test)
            # # muNN2test = 0.01*act_gauss(in_mu2test)
            # muNN2test = 0.05 * act_gauss(in_mu2test)

            dS2dt = tf.matmul(Amat[0:-1, :], S_observe)
            dI2dt = tf.matmul(Amat[0:-1, :], I_observe)
            dR2dt = tf.matmul(Amat[0:-1, :], R_observe)
            dD2dt = tf.matmul(Amat[0:-1, :], D_observe)

            temp_s2t = -betaNN2train[0:-1, 0] * S_observe[0:-1, 0] * I_observe[0:-1, 0] / (S_observe[0:-1, 0] + I_observe[0:-1, 0])
            temp_i2t = betaNN2train[0:-1, 0] * S_observe[0:-1, 0] * I_observe[0:-1, 0] / (S_observe[0:-1, 0] + I_observe[0:-1, 0]) - \
                       gammaNN2train[0:-1, 0] * I_observe[0:-1, 0] - muNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_r2t = gammaNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_d2t = muNN2train[0:-1, 0] * I_observe[0:-1, 0]

            if params['loss_function'].lower() == 'l2_loss':
                # Loss2dS = tf.reduce_mean(tf.square(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1])))
                Loss2dI = tf.reduce_mean(tf.square(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1])))
                Loss2dR = tf.reduce_mean(tf.square(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1])))
                Loss2dD = tf.reduce_mean(tf.square(dD2dt - tf.reshape(temp_d2t, shape=[-1, 1])))
            elif params['loss_function'].lower() == 'lncosh_loss':
                # Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1]))))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1]))))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1]))))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD2dt - tf.reshape(temp_d2t, shape=[-1, 1]))))

            # 正则化
            regular_func = lambda a, b: tf.constant(0.0)
            if FLAGS.regular_method == 'L1':
                regular_func = modelUtils.regular_weights_biases_L1
            elif FLAGS.regular_method == 'L2':
                regular_func = modelUtils.regular_weights_biases_L2
            regular_WB2Beta = regular_func(Weight2beta, Bias2beta)
            regular_WB2Gamma = regular_func(Weight2gamma, Bias2gamma)
            regular_WB2Mu = regular_func(Weight2mu, Bias2mu) 

            PWB2Beta = wb_penalty * regular_WB2Beta
            PWB2Gamma = wb_penalty * regular_WB2Gamma
            PWB2Mu = wb_penalty * regular_WB2Mu

            # Loss = Loss2dS + Loss2dI + Loss2dR + Loss2dD + PWB2Beta + PWB2Gamma + PWB2Mu
            Loss =  Loss2dI + Loss2dR + Loss2dD + PWB2Beta + PWB2Gamma + PWB2Mu

            # optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            # train_Losses = optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    # loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], [], []
    loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], []
    lr_list = list()
    test_epoch = []

    filename = './data/minnesota3.csv'
    date, s_data, i_data, r_data, d_data = dataUtils.load_csvdata2(filename, N=3450000, aveFlag=True,normalizeFlag=False)

    train_data, test_data = dataUtils.split_data(date, s_data, i_data, r_data, d_data, train_size=0.75)  
    train_date, train_data2s, train_data2i, train_data2r, train_data2d, *_ = train_data

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行

    lr_list = list()
    # 通过exponential_decay函数生成学习率
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate = FLAGS.initial_learning_rate,
                                                        global_step = global_steps,
                                                        decay_steps = FLAGS.decay_steps,
                                                        decay_rate = FLAGS.decay_rate,
                                                        staircase = False)
    # 优化器
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_Losses = optimizer.minimize(Loss, global_step=global_steps)

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # tmp_lr = init_lr
        for epoch in range(FLAGS.train_epoches):
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                dataUtils.sample_data(train_date, train_data2s, train_data2i, train_data2r, train_data2d,
                                        window_size=batchSize_train, sampling_opt=params['opt2sample'])
            
            # _, loss_s, loss_i, loss_r, loss_d, loss, pwb2beta, pwb2gamma, pwb2mu, r2t, R2dt = sess.run(
            #     [train_Losses, Loss2dS, Loss2dI, Loss2dR, Loss2dD, Loss, PWB2Beta, PWB2Gamma, PWB2Mu, temp_r2t, dR2dt],
            #     feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})

            _, loss_i, loss_r, loss_d, loss, pwb2beta, pwb2gamma, pwb2mu, r2t, R2dt = sess.run(
                [train_Losses, Loss2dI, Loss2dR, Loss2dD, Loss, PWB2Beta, PWB2Gamma, PWB2Mu, temp_r2t, dR2dt],
                feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})

            s2t, tmp_s2t, i2t, tmp_i2t,r2t, tmp_r2t,d2t, tmp_d2t, = sess.run(
                [dS2dt, temp_s2t, dI2dt, temp_i2t, dR2dt, temp_r2t,dD2dt, temp_d2t],
                feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})            
            
            lr = sess.run(learning_rate)
            lr_list.append(sess.run(learning_rate))  

            # loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_all.append(loss)

            if epoch % 1000 == 0:
                print_and_log2train(epoch, time.time() - t0, lr, pwb2beta, pwb2gamma, pwb2mu,
                                    loss_i, loss_r, loss_d, loss, log_out=log_fileout)
        
                # 以下代码为输出训练过程中 beta, gamma, mu 的训练结果
                test_epoch.append(epoch / 1000)
                beta2test, gamma2test, mu2test = sess.run([betaNN2train, gammaNN2train, muNN2train],
                                                             feed_dict={T_train: np.reshape(date, [-1, 1])})

                ds_dt, pre_s2t, di_dt, pre_i2t, dr_dt, pre_r2t, dd_dt, pre_d2t, = sess.run(
                    [dS2dt, temp_s2t, dI2dt, temp_i2t, dR2dt, temp_r2t,dD2dt, temp_d2t],
                    feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs,
                            learning_rate: lr})

        data_dic = {'ds_dt': np.squeeze(ds_dt),'pre_s2t': np.squeeze(pre_s2t),
                    'di_dt': np.squeeze(di_dt),'pre_i2t': np.squeeze(pre_i2t),
                    'dr_dt': np.squeeze(dr_dt),'pre_r2t': np.squeeze(pre_r2t),
                    'dd_dt': np.squeeze(dd_dt),'pre_d2t': np.squeeze(pre_d2t)
        }
        sird_df = pd.DataFrame.from_dict(data_dic)
        sird_df.to_csv(params['FolderName'] + '/sird_results.csv', index = False)     

        paras_dic = {'beta2test': np.squeeze(beta2test), 
                     'gamma2test': np.squeeze(gamma2test), 
                     'mu2test': np.squeeze(mu2test)}
        paras_df = pd.DataFrame.from_dict(paras_dic)
        paras_df.to_csv(params['FolderName'] + '/params_results.csv', index = False)

        # save loss data
        loss_dic = {#'loss_s':loss_s_all,
                    'loss_i':loss_i_all,
                    'loss_r':loss_r_all,
                    'loss_d':loss_d_all,
                    'lr': lr_list}
        
        loss_df = pd.DataFrame.from_dict(loss_dic)
        loss_df.to_csv(params['FolderName'] + '/loss_results.csv', index = False)

        # saveData.save_trainParas2mat_Covid(beta2train, name2para='beta2train', outPath=params['FolderName'])
        # saveData.save_trainParas2mat_Covid(gamma2train, name2para='gamma2train', outPath=params['FolderName'])
        # saveData.save_trainParas2mat_Covid(mu2train, name2para='mu2train', outPath=params['FolderName'])

        # plotData.plot_Para2convid(beta2train, name2para='beta_train',
        #                           coord_points2test=np.reshape(train_date, [-1, 1]), outPath=params['FolderName'])
        # plotData.plot_Para2convid(gamma2train, name2para='gamma_train',
        #                           coord_points2test=np.reshape(train_date, [-1, 1]), outPath=params['FolderName'])
        # plotData.plot_Para2convid(mu2train, name2para='mu_train',
        #                           coord_points2test=np.reshape(train_date, [-1, 1]), outPath=params['FolderName'])

        # saveData.save_SIRD_trainLoss2mat_no_N(loss_s_all, loss_i_all, loss_r_all, loss_d_all, actName=act_func2paras,
        #                                       outPath=params['FolderName'])

        # plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=params['seed'], outPath=params['FolderName'],
        #                                   yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss2d', seedNo=params['seed'], outPath=params['FolderName'],
                                          yaxis_scale=True)

        # saveData.save_SIRD_testParas2mat(beta2test, gamma2test, mu2test, name2para1='beta2test',
        #                                  name2para2='gamma2test', name2para3='mu2test', outPath=params['FolderName'])

        # plotData.plot_Para2convid(beta2test, name2para='beta_test', coord_points2test=test_t_bach,
        #                           outPath=params['FolderName'])
        # plotData.plot_Para2convid(gamma2test, name2para='gamma_test', coord_points2test=test_t_bach,
        #                           outPath=params['FolderName'])
        # plotData.plot_Para2convid(mu2test, name2para='mu_test', coord_points2test=test_t_bach,
        #                           outPath=params['FolderName'])


def main(unused_argv):
    params = {}
    params['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'output_SIRD'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    params['seed'] = 43
    timeFolder = datetime.now().strftime("%Y%m%d_%H%M")       # 当前时间为文件夹名
    params['FolderName'] = os.path.join(OUT_DIR, timeFolder)  # 路径连接
    FolderName = params['FolderName']
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # params['activate_stop'] = int(step_stop_flag)
    params['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    params['max_epoch'] = 60000
    if 0 != params['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        params['max_epoch'] = int(epoch_stop)

    # ----------------------------------------- Convid 设置 ---------------------------------
    params['eqs_name'] = 'SIRD'
    # params['input_dim'] = 1                       # 输入维数，即问题的维数(几元问题)
    # params['output_dim'] = 1                      # 输出维数
    # params['total_population'] = 3450000          # 总的“人口”数量

    # ------------------------------------  神经网络的设置  ----------------------------------------
    # params['size2train'] = 250                    # 训练集的大小
    # params['batch_size2train'] = 30              # 训练数据的批大小
    # params['batch_size2test'] = 50               # 训练数据的批大小
    # params['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    # params['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序
    params['opt2sample'] = 'sequential_sort'  # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    # params['regular_weight_model'] = 'L1'
    params['regular_weight_model'] = 'L2'          # The model of regular weights and biases
    params['regular_weight'] = 0.001             # Regularization parameter for weights
    # params['regular_weight'] = 0.0005            # Regularization parameter for weights
    # params['regular_weight'] = 0.0001            # Regularization parameter for weights
    # params['regular_weight'] = 0.00005             # Regularization parameter for weights
    # params['regular_weight'] = 0.00001           # Regularization parameter for weights

    params['optimizer_name'] = 'Adam'              # 优化器
    params['loss_function'] = 'L2_loss'            # 损失函数的类型
    # params['loss_function'] = 'lncosh_loss'      # 损失函数的类型
    params['scale_up'] = 1                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    params['scale_factor'] = 100                   # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    params['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < params['max_epoch']:
        params['learning_rate'] = 2e-3             # 学习率
        params['lr_decay'] = 1e-4                  # 学习率 decay
        # params['learning_rate'] = 2e-4           # 学习率
        # params['lr_decay'] = 5e-5                # 学习率 decay
    elif (20000 < params['max_epoch'] and 50000 >= params['max_epoch']):
        # params['learning_rate'] = 1e-3           # 学习率
        # params['lr_decay'] = 1e-4                # 学习率 decay
        # params['learning_rate'] = 2e-4           # 学习率
        # params['lr_decay'] = 1e-4                # 学习率 decay
        params['learning_rate'] = 1e-4             # 学习率
        params['lr_decay'] = 5e-5                  # 学习率 decay
    else:
        params['learning_rate'] = 5e-5             # 学习率
        params['lr_decay'] = 1e-5                  # 学习率 decay

    # SIRD参数网络模型的选择
    # params['model2paras'] = 'DNN'
    # params['model2paras'] = 'DNN_scale'
    # params['model2paras'] = 'DNN_scaleOut'
    params['model2paras'] = 'DNN_FourierBase'

    # SIRD参数网络模型的隐藏层单元数目
    if params['model2paras'] == 'DNN_FourierBase':
        params['hidden2para'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # params['hidden2para'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        params['hidden2para'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # params['hidden2para'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # params['hidden2para'] = (100, 100, 80, 60, 60, 40)
        # params['hidden2para'] = (200, 100, 100, 80, 50, 50)

    # SIRD参数网络模型的尺度因子
    if params['model2paras'] != 'DNN':
        params['freq2paras'] = np.concatenate(([1], np.arange(1, 25)), axis=0)

    # SIRD参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    if params['model2paras'] == 'DNN_FourierBase' or params['model2paras'] == 'DNN_scale':
        params['if_repeat_High_freq2paras'] = False

    # SIRD参数网络模型的激活函数的选择
    # params['actIn_Name2paras'] = 'relu'
    # params['actIn_Name2paras'] = 'leaky_relu'
    # params['actIn_Name2paras'] = 'sigmoid'
    # params['actIn_Name2paras'] = 'tanh'
    # params['actIn_Name2paras'] = 'srelu'
    # params['actIn_Name2paras'] = 's2relu'
    params['actIn_Name2paras'] = 'sin'
    # params['actIn_Name2paras'] = 'sinAddcod'
    # params['actIn_Name2paras'] = 'elu'
    # params['actIn_Name2paras'] = 'gelu'
    # params['actIn_Name2paras'] = 'mgelu'
    # params['actIn_Name2paras'] = 'linear'

    # params['act_Name2paras'] = 'relu'
    # params['act_Name2paras'] = 'leaky_relu'
    # params['act_Name2paras'] = 'sigmoid'
    # params['act_Name2paras'] = 'tanh'  # 这个激活函数比较s2ReLU合适
    # params['act_Name2paras'] = 'srelu'
    # params['act_Name2paras'] = 's2relu'
    params['act_Name2paras'] = 'sin'
    # params['act_Name2paras'] = 'sinAddcos'
    # params['act_Name2paras'] = 'elu'
    # params['act_Name2paras'] = 'gelu'
    # params['act_Name2paras'] = 'mgelu'
    # params['act_Name2paras'] = 'linear'

    solve_SIRD2COVID(params)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)