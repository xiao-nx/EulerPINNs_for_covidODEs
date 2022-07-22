import os
import sys
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# def load_csvdata(fileName, N=3450000, normalizeFlag=True):
#     '''
#     fileName: path of csv file
#     N: total population in the city(country)
#     normalizeFlag: whether normalize the data to 0-1.
#     '''
#     df = pd.read_csv(fileName)
#     date_list = np.array(df['date'])
#     date_list = [(i+1) for i in range(len(df))]
#     df['current_confiremed'] = df['cum_confirmed'] - df['recovered'] - df['death'] # 对应公式里的I
#     infective_list = np.array(df['current_confiremed'])
#     recovery_list = np.array(df['recovered'])
#     death_list = np.array(df['death'])
#     df['pops'] = N - df['current_confiremed'] - df['recovered'] - df['death']
#     susceptible_list = np.array(df['pops'])
#     data_list = [date_list, susceptible_list, infective_list, recovery_list, death_list]

#     if normalizeFlag == True:
#         data_list = [[item / int(N) for item in sublist] for sublist in data_list]

#     return data_list

def load_csvdata(fileName, N=3450000, filter_size=5, normalizeFlag=False):
    '''
    fileName: path of csv file
    N: total population in the city(country)
    normalizeFlag: whether normalize the data to 0-1.
    '''
    df = pd.read_csv(fileName)
    # date_list = np.array(df['date'])
    # date_list = [(i+1) for i in range(len(df))]
    df['current_confiremed'] = df['cum_confirmed'] - df['recovered'] - df['death'] # 对应公式里的I
    # df['pops'] = N - df['current_confiremed'] - df['recovered'] - df['death']
    df['pops'] = N - df['cum_confirmed']

    if filter_size != 0:
        df['pops'] = df['pops'].rolling(window=filter_size).mean()
        df['current_confiremed'] = df['current_confiremed'].rolling(window=filter_size).mean()
        df['recovered'] = df['recovered'].rolling(window=filter_size).mean()
        df['death'] = df['death'].rolling(window=filter_size).mean()

    df = df.dropna(axis=0)

    susceptible_list = np.array(df['pops'])
    infective_list = np.array(df['current_confiremed'])
    recovery_list = np.array(df['recovered'])
    death_list = np.array(df['death'])

    date_list = [(i+1) for i in range(len(death_list))]
    
    data_list = [date_list, susceptible_list, infective_list, recovery_list, death_list]

    if normalizeFlag == True:
        data_list = [[item / int(N) for item in sublist] for sublist in data_list]

    return data_list

# 拆分数据为训练集和测试集
def split_data(data_list, train_size=0.75):
    date_list, susceptible_list, infective_list, recovery_list, death_list, *_ = data_list
    train_size = int(len(date_list) * train_size)
    # test = len(date_list) - train_size

    date_train = date_list[0:train_size]
    susceptible_train = susceptible_list[0:train_size]
    infective_train = infective_list[0: train_size]
    recovery_train = recovery_list[0:train_size]
    death_train = death_list[0:train_size]

    date_test = date_list[train_size:-1]
    susceptible_test = susceptible_list[train_size:-1]
    infective_test = infective_list[train_size:-1]
    recovery_test = recovery_list[train_size:-1] 
    death_test = death_list[train_size:-1]

    train_data = [date_train, susceptible_train, infective_train, recovery_train, death_train]
    test_data = [date_test, susceptible_test, infective_test, recovery_test, death_test]

    return train_data, test_data


def window_sample(data_list, window_size=7, method='sequential_sort'):
    date_data, s_data, i_data,r_data, d_data, *_ = data_list
    data_length = len(date_data)
    assert method in ['random', 'random_sort', 'sequential_sort']
    if method == 'random':
        indexes = np.random.randint(data_length, size=window_size)
    elif method == 'random_sort':
        indexes_temp = np.random.randint(data_length, size=window_size)
        indexes = np.sort(indexes_temp)
    elif method == 'sequential_sort':
        index_base = np.random.randint(data_length - window_size, size=1)
        indexes = np.arange(index_base, index_base + window_size)

    date_window = [date_data[idx] for idx in indexes]
    s_window = [s_data[idx] for idx in indexes]
    i_window = [i_data[idx] for idx in indexes]
    r_window = [r_data[idx] for idx in indexes]
    d_window = [d_data[idx] for idx in indexes]

    date_window = np.expand_dims(date_window, axis=1)
    s_window = np.expand_dims(s_window, axis=1)
    i_window = np.expand_dims(i_window, axis=1)
    r_window = np.expand_dims(r_window, axis=1)
    d_window = np.expand_dims(d_window, axis=1)

    data_window = [date_window, s_window, i_window, r_window, d_window]

    return data_window



def generate_dataset(df, column_name, time_step=7):
    
    data = df[[column_name]] 
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(size=time_step, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(time_step))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

    return dataset


# def load_csvdata(fileName, N=3450000):
#     '''
#     fileName: path of csv file
#     N: total population in the city(country)
#     '''
#     df = pd.read_csv(fileName)
#     date_list = [(i + 1) for i in range(len(df))]
#     df['current_confiremed'] = df['cum_confirmed'] - df['recovered'] - df['death'] # 对应公式里的I
#     infective_list = np.array(df['current_confiremed'])
#     recovery_list = np.array(df['recovered'])
#     death_list = np.array(df['death'])
#     df['pops'] = N - df['current_confiremed'] - df['recovered'] - df['death']
#     susceptible_list = np.array(df['pops'])
#     data_list = [date_list, susceptible_list, infective_list, recovery_list, death_list]

#     return data_list

def windowed_dataset(data_list, time_step=7, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.window(size=time_step + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(time_step + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

def sliding_window(data_list, window_size=14, step_time=7):
    # date, s_data, i_data, r_data, d_data, *_ = data_list
    data_list = np.array(data_list)
    length = len(data_list[0]) - window_size
    for idx in range(0, length, step_time):
        start = idx 
        end = start + window_size
        window_data = data_list[:,start: end]
        yield window_data



# 从总体数据集中载入部分数据作为训练集
# 窗口滑动采样时，batch size = window_size 
def sample_data(date_data, s_data, i_data,r_data, d_data, window_size=1, sampling_opt=None):
    date_temp = list()
    s_temp = list()
    i_temp = list()
    r_temp = list()
    d_temp = list()
    data_length = len(date_data)
    if sampling_opt.lower() == 'random_sample':
        indexes = np.random.randint(data_length, size=window_size)
    elif sampling_opt.lower() == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=window_size)
        indexes = np.sort(indexes_temp)
    elif sampling_opt.lower() == 'sequential_sort':
        index_base = np.random.randint(data_length - window_size, size=1)
        indexes = np.arange(index_base, index_base + window_size)
    else:
        print('woring!!!!')
    for i_index in indexes:
        date_temp.append(float(date_data[i_index]))
        s_temp.append(float(s_data[i_index]))
        i_temp.append(float(i_data[i_index]))
        r_temp.append(float(r_data[i_index]))
        d_temp.append(float(d_data[i_index]))

    date_samples = np.array(date_temp)
    # data_samples = np.array(data_temp)
    s_samples = np.array(s_temp)
    i_samples = np.array(i_temp)
    r_samples = np.array(r_temp)
    d_samples = np.array(d_temp)
    date_samples = date_samples.reshape(window_size, 1)
    # data_samples = data_samples.reshape(batchsize, 1)
    s_samples = s_samples.reshape(window_size, 1)
    i_samples = i_samples.reshape(window_size, 1)
    r_samples = r_samples.reshape(window_size, 1)
    d_samples = d_samples.reshape(window_size, 1)

    return date_samples, s_samples, i_samples, r_samples, d_samples



# def window_sample(data_list, window_size=7):
#     date, s_data, i_data, r_data, d_data, *_ = data_list
#     for i in range(len(date) - window_size):
#         start = i 
#         end = i + window_size
#         window_data = [date[start: end], s_data[start: end], i_data[start: end], r_data[start: end], d_data[start: end]]
#         yield window_data





# 还是要重新写采样函数的
# overlap + 保证一次运行完所有数据

def compute_mse_res(data_obs, nn_predict):
    point_ERR2I = np.square(nn_predict - data_obs)
    mse = np.mean(point_ERR2I)
    res = mse / np.mean(np.square(data_obs))

    return mse, res

# point_ERR2I = np.square(i_nn2test - i_obs_test)
# test_mse2I = np.mean(point_ERR2I)
# test_mse2I_all.append(test_mse2I)
# test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
# test_rel2I_all.append(test_rel2I)