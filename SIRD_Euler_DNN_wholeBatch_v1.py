"""
@author: Xiao Ning
Benchmark Code of SIRD model
2022-07-12
"""
import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
import platform
import shutil

import argparse
from sre_parse import FLAGS

import plotData
import dataUtils
import logUtils
import modelUtils

import tensorflow as tf
print('tensorflow version: \n',tf.version.VERSION)
print (tf.executing_eagerly())
# tf.compat.v1.enable_eager_execution()

# tf2兼容tf1
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('--data_fname', type=str, default='./data/minnesota_0719.csv',
                    help='data path')

parser.add_argument('--population', type=float, default=3450000,
                    help='.')

parser.add_argument('--input_dim', type=float, default=1,
                    help='.')

parser.add_argument('--output_dim', type=float, default=1,
                    help='.')

parser.add_argument('--output_dir', type=str, default='./output_SIRD_Euler',
                    help='data path')

parser.add_argument('--train_epoches', type=int, default=100000,
                    help='max epoch during training.')

parser.add_argument('--eval_epoches', type=int, default=2000, 
                    help='The number of training epochs to run between evaluations.') 

parser.add_argument('--gpuNumber', type=int, default=0, 
                    help='The number of training epochs to run between evaluations.') 

parser.add_argument('--optimizer', type=str, default='Adam', 
                    help='optimizer')

parser.add_argument('--train_loss', type=str, default='train_group', 
                    help='loss')

parser.add_argument('--filter_size', type=int, default=5, 
                    help='filter data for training')

# parser.add_argument('--batch_size', type=int, default=16,
#                     help='.')

# parser.add_argument('--sample_method', type=str, default='sequential_sort',
#                     help='. : random_sample, rand_sample_sort, sequential_sort')

parser.add_argument('--params_network', type=str, default='DNN_FOURIERBASE',
                    help='network archtecture:' 'DNN, DNN_FOURIERBASE, DNN_SCALE')

parser.add_argument('--hidden_params', type=set, default=([35, 50, 30, 30, 20]),
                    help='hidden layers:'
                         '(80, 80, 60, 40, 40, 20)'
                         '(100, 100, 80, 60, 60, 40)'
                         '(200, 100, 100, 80, 50, 50)')  

parser.add_argument('--loss_function', type=str, default='L2_loss',
                    help='loss function:' 'L2_loss, lncosh_loss')   

parser.add_argument('--input_active', type=str, default='tanh', # sin效果也不错
                    help='activate function for input layer:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')  
parser.add_argument('--hidden_active', type=str, default='tanh', # sin效果也不错
                    help='activate function:' 'tanh, relu, srelu, s2relu, leaky_relu, elu, selu, phi')                                     

parser.add_argument('--initial_learning_rate', type=float, default=0.005,
                    help='.'
                    '0.1, 0.01, 0.05, 0,001')
parser.add_argument('--decay_steps', type=float, default=1000,
                    help='.' '0.1, 0.01, 0.05, 0,001')
parser.add_argument('--decay_rate', type=float, default=0.95,
                    help='.' '0.1, 0.01, 0.05, 0,001')

parser.add_argument('--regular_method', type=str, default='L2',
                    help='The method of regular weights and biases:' 'L0, L1')

parser.add_argument('--regular_weight', type=float, default=0.00005, # 神经网络参数的惩罚因子
                    help='Regularization parameter for weights.' '0.00001, 0.00005, 0.0001, 0.0005, 0.001')

def loss_func(dS2dt_obs, dI2dt_obs, dR2dt_obs, dD2dt_obs, temp_s2t, temp_i2t, temp_r2t, temp_d2t,regular_WB2Beta, regular_WB2Gamma, regular_WB2Mu):
    wb_penalty = FLAGS.regular_weight  # 神经网络参数的惩罚因子
    if FLAGS.loss_function == 'L2_loss':
        Loss2dS = tf.reduce_mean(tf.square(dS2dt_obs - tf.reshape(temp_s2t, shape=[-1, 1])))
        Loss2dI = tf.reduce_mean(tf.square(dI2dt_obs - tf.reshape(temp_i2t, shape=[-1, 1])))
        Loss2dR = tf.reduce_mean(tf.square(dR2dt_obs - tf.reshape(temp_r2t, shape=[-1, 1])))
        Loss2dD = tf.reduce_mean(tf.square(dD2dt_obs - tf.reshape(temp_d2t, shape=[-1, 1])))
    elif FLAGS.loss_function == 'lncosh_loss':
        Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS2dt_obs - tf.reshape(temp_s2t, shape=[-1, 1]))))
        Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI2dt_obs - tf.reshape(temp_i2t, shape=[-1, 1]))))
        Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR2dt_obs - tf.reshape(temp_r2t, shape=[-1, 1]))))
        Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD2dt_obs - tf.reshape(temp_d2t, shape=[-1, 1]))))

    PWB2Beta = wb_penalty * regular_WB2Beta
    PWB2Gamma = wb_penalty * regular_WB2Gamma
    PWB2Mu = wb_penalty * regular_WB2Mu

    Loss = Loss2dS + Loss2dI + Loss2dR + Loss2dD + PWB2Beta + PWB2Gamma + PWB2Mu

    return Loss, Loss2dS, Loss2dI, Loss2dR, Loss2dD, PWB2Beta, PWB2Gamma, PWB2Mu

def solve_SIRD2COVID(FolderName):

    log_fileout = open(os.path.join(FolderName, 'log_train.txt'), 'a+')  

    filename = FLAGS.data_fname
    data_list = dataUtils.load_csvdata(filename, N=FLAGS.population, filter_size=FLAGS.filter_size,normalizeFlag=False)
    batchSize_train = len(data_list[0])

    input_dim = FLAGS.input_dim
    out_dim = FLAGS.output_dim

    input_active = FLAGS.input_active
    hidden_active = FLAGS.hidden_active

    regular_method = FLAGS.regular_method

    T_train = tf.compat.v1.placeholder(tf.float32, name='T_train', shape=[None, out_dim])
    S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, out_dim])
    I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
    R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[None, out_dim])
    D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[None, out_dim])

    data_matrix = modelUtils.EulerIteration(batchSize_train, S_observe, I_observe, R_observe, D_observe)
    dS2dt_obs, dI2dt_obs, dR2dt_obs, dD2dt_obs = data_matrix['dS2dt'], data_matrix['dI2dt'], data_matrix['dR2dt'], data_matrix['dD2dt']
    
    # 选择网络结构
    if FLAGS.params_network == 'DNN':
        betaNN2train, gammaNN2train, muNN2train, regular_WB2Beta, regular_WB2Gamma, regular_WB2Mu = modelUtils.DNN_networks(T_train,input_dim, out_dim,input_active, hidden_active, regular_method)
    elif FLAGS.params_network == 'DNN_SCALE':
        betaNN2train, gammaNN2train, muNN2train, regular_WB2Beta, regular_WB2Gamma, regular_WB2Mu = modelUtils.scale_dnn_networks(T_train, input_dim, out_dim,input_active, hidden_active, regular_method)
    elif FLAGS.params_network == 'DNN_FOURIERBASE':
        betaNN2train, gammaNN2train, muNN2train, regular_WB2Beta, regular_WB2Gamma, regular_WB2Mu = modelUtils.Fourier_dnn_networks(T_train,input_dim, out_dim, hidden_active, regular_method)
    
    temp_s2t, temp_i2t, temp_r2t, temp_d2t = modelUtils.SIRD_model(betaNN2train, gammaNN2train, muNN2train, S_observe, I_observe)

    Loss, Loss2dS, Loss2dI, Loss2dR, Loss2dD, PWB2Beta, PWB2Gamma, PWB2Mu = loss_func(dS2dt_obs, dI2dt_obs, dR2dt_obs, dD2dt_obs, \
                                                                                      temp_s2t, temp_i2t, temp_r2t, temp_d2t,\
                                                                                      regular_WB2Beta, regular_WB2Gamma, regular_WB2Mu)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], [], []
    lr_list = list()

    # filename = FLAGS.data_fname
    # data_list = dataUtils.load_csvdata2(filename, N=3450000, aveFlag=True,normalizeFlag=False)

    # 是否拆分数据 （不拆了）
    # train_data, test_data = dataUtils.split_data(data_list, train_size=1.0) 
    train_date, train_data2s, train_data2i, train_data2r, train_data2d, *_ = data_list
    date = data_list[0]

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行

    global_steps = tf.Variable(0, trainable=False)
    # 通过exponential_decay函数生成学习率
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate = FLAGS.initial_learning_rate,
                                                        global_step = global_steps,
                                                        decay_steps = FLAGS.decay_steps,
                                                        decay_rate = FLAGS.decay_rate,
                                                        staircase = False)
    lr_list = list()
    # 优化器
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    # train_Losses = optimizer.minimize(Loss, global_step=global_steps)

    if FLAGS.train_loss == 'train_group':
        train_Loss2S = optimizer.minimize(Loss2dS, global_step=global_steps)
        train_Loss2I = optimizer.minimize(Loss2dI, global_step=global_steps)
        train_Loss2R = optimizer.minimize(Loss2dR, global_step=global_steps)
        train_Loss2D = optimizer.minimize(Loss2dD, global_step=global_steps)
        train_Losses = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2D)
    elif FLAGS.train_loss == 'train_union_loss':
        train_Losses = optimizer.minimize(Loss, global_step=global_steps)

    with tf.device('/gpu:%s' % (FLAGS.gpuNumber)):
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(FLAGS.train_epoches + 1):
                t_batch = np.expand_dims(train_date, axis=1)
                s_obs = np.expand_dims(train_data2s, axis=1)
                i_obs = np.expand_dims(train_data2i, axis=1)
                r_obs = np.expand_dims(train_data2r, axis=1)
                d_obs = np.expand_dims(train_data2d, axis=1)

                _, loss_s, loss_i, loss_r, loss_d, loss, pwb2beta, pwb2gamma, pwb2mu = sess.run(
                    [train_Losses, Loss2dS, Loss2dI, Loss2dR, Loss2dD, Loss, PWB2Beta, PWB2Gamma, PWB2Mu],
                    feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})

                s2t, tmp_s2t, i2t, tmp_i2t,r2t, tmp_r2t, d2t, tmp_d2t, = sess.run(
                    [dS2dt_obs, temp_s2t, dI2dt_obs, temp_i2t, dR2dt_obs, temp_r2t,dD2dt_obs, temp_d2t],
                    feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs})            
                
                lr = sess.run(learning_rate)
                lr_list.append(sess.run(learning_rate))  

                loss_s_all.append(loss_s)
                loss_i_all.append(loss_i)
                loss_r_all.append(loss_r)
                loss_d_all.append(loss_d)
                loss_all.append(loss)

                if epoch % 1000 == 0:
                    logUtils.print_training(epoch, time.time() - t0, lr, pwb2beta, pwb2gamma, pwb2mu, loss_s,
                                        loss_i, loss_r, loss_d, loss, log_out=log_fileout)
            
                    # 以下代码为输出训练过程中 beta, gamma, mu 的训练结果
                    beta2test, gamma2test, mu2test = sess.run([betaNN2train, gammaNN2train, muNN2train],
                                                                feed_dict={T_train: np.reshape(date, [-1, 1])})
                    ds_dt, pre_s2t, di_dt, pre_i2t, dr_dt, pre_r2t, dd_dt, pre_d2t, = sess.run(
                        [dS2dt_obs, temp_s2t, dI2dt_obs, temp_i2t, dR2dt_obs, temp_r2t,dD2dt_obs, temp_d2t],
                        feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs,
                                learning_rate: lr})

            # sav data
            data_dic = {'ds_dt': np.squeeze(ds_dt),'pre_s2t': np.squeeze(pre_s2t),
                        'di_dt': np.squeeze(di_dt),'pre_i2t': np.squeeze(pre_i2t),
                        'dr_dt': np.squeeze(dr_dt),'pre_r2t': np.squeeze(pre_r2t),
                        'dd_dt': np.squeeze(dd_dt),'pre_d2t': np.squeeze(pre_d2t)
            }
            sird_df = pd.DataFrame.from_dict(data_dic)
            sird_df.to_csv(FolderName + '/sird_results.csv', index = False)     

            paras_dic = {'beta2test': np.squeeze(beta2test), 
                        'gamma2test': np.squeeze(gamma2test), 
                        'mu2test': np.squeeze(mu2test)}
            paras_df = pd.DataFrame.from_dict(paras_dic)
            paras_df.to_csv(FolderName + '/params_results.csv', index = False)

            # save loss data
            loss_dic = {'loss_s':loss_s_all,
                        'loss_i':loss_i_all,
                        'loss_r':loss_r_all,
                        'loss_d':loss_d_all,
                        'lr': lr_list}
            
            loss_df = pd.DataFrame.from_dict(loss_dic)
            loss_df.to_csv(FolderName + '/loss_results.csv', index = False)

            # plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', outPath=FolderName,
            #                                 yaxis_scale=True)
            # plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', outPath=FolderName,
            #                                 yaxis_scale=True)
            # plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', outPath=FolderName,
            #                                 yaxis_scale=True)
            # plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss2d', outPath=FolderName,
            #                                 yaxis_scale=True)

def main(unused_argv):

    # 文件保存路径设置
    store_file = FLAGS.output_dir
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    timeFolder = datetime.now().strftime("%Y%m%d_%H%M")       # 当前时间为文件夹名
    FolderName = os.path.join(OUT_DIR, timeFolder)  # 路径连接
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)

    # 复制并保存当前文件
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    args = parser.parse_args()
    dict = vars(args)

    log_fileout = os.path.join(FolderName, './log_train.txt')   
    with open(log_fileout, 'w') as f:
        f.write('related parameters:  \n')
        for key in dict:
            f.write(str(key) + ":  " + str(dict[key]) + '\n')
        f.write('===========================================\n\n')

    solve_SIRD2COVID(FolderName)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)


 