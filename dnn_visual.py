"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/11/19 13:31
"""
import os
import numpy as np
import h5py, librosa
from visual_api import *
from file_operating import *
from audio_processing import *
from scipy import signal
import matplotlib.pylab as plt

def get_parameter(model_dir, key):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if(os.path.splitext(file)[-1].lower() == '.meta'):
                ckpt = file
    ckpt_path = model_dir + ckpt.split('.')[0]
    reader = tf.train.NewCheckpointReader(ckpt_path)
    all_variables = reader.get_variable_to_shape_map()
    print(all_variables)
    data = reader.get_tensor(key)
    return data

def scale_feature(data, mean, std):
    for i in range(data.shape[0]):
        if len(data.shape) == 3:
            for j in range(data.shape[1]):
                data[i, j, :] = (data[i, j, :] - mean) / std
        if len(data.shape) == 2:
            data[i, :] = (data[i, :] - mean) / std
    return data

def model_build(x):
    model_dir = 'C:/Users/asus/Desktop/dnn_demo/mini_model/'
    time_size = 2
    freq_size = 3
    rnn_size = 64
    rnn_num_layers = 2
    x_shape = tf.shape(x)
    n_feature = x_shape[0]
    n_height = 3
    strides = [1, 1, 2, 1]

    # init Session
    conv2_1 = conv2d_layer(x, get_parameter(model_dir, 'generator_model/cv1/w'), get_parameter(model_dir, 'generator_model/cv1/b'),
                           strides=strides, activation_function_type='lrelu', padding='SAME')
    conv2_2 = conv2d_layer(conv2_1, get_parameter(model_dir, 'generator_model/cv2/w'), get_parameter(model_dir, 'generator_model/cv2/b'),
                           strides=strides, activation_function_type='lrelu')
    conv2_3 = conv2d_layer(conv2_2, get_parameter(model_dir, 'generator_model/cv3/w'), get_parameter(model_dir, 'generator_model/cv3/b'),
                           strides=strides, activation_function_type='lrelu')
    conv2_4 = conv2d_layer(conv2_3, get_parameter(model_dir, 'generator_model/cv4/w'), get_parameter(model_dir, 'generator_model/cv4/b'),
                           strides=strides, activation_function_type='lrelu')
    conv2_5 = conv2d_layer(conv2_4,get_parameter(model_dir, 'generator_model/cv5/w'), get_parameter(model_dir, 'generator_model/cv5/b'),
                           strides=strides, activation_function_type='lrelu')
    conv2_6 = conv2d_layer(conv2_5, get_parameter(model_dir, 'generator_model/cv6/w'), get_parameter(model_dir, 'generator_model/cv6/b'),
                           strides=strides, activation_function_type='lrelu', padding='VALID')
    reshape_1 = tf.reshape(conv2_6, [-1, 3, 4 * 16])
    reshape_1 = tf.cast(reshape_1, dtype=tf.float32)

    multi_rnn_cells = [get_parameter(model_dir,'generator_model/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel'),
                       get_parameter(model_dir, 'generator_model/rnn/multi_rnn_cell/cell_0/lstm_cell/bias'),
                       get_parameter(model_dir, 'generator_model/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel'),
                       get_parameter(model_dir, 'generator_model/rnn/multi_rnn_cell/cell_1/lstm_cell/bias')]
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(64,initializer=tf.constant_initializer(multi_rnn_cells[0], dtype=tf.float32)),
                  tf.nn.rnn_cell.LSTMCell(64, initializer=tf.constant_initializer(multi_rnn_cells[2], dtype=tf.float32))]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=reshape_1, dtype=tf.float32)

    tv = [v.name for v in tf.trainable_variables()]
    with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
        a = tf.get_variable('multi_rnn_cell/cell_0/lstm_cell/bias', shape=multi_rnn_cells[1].shape)
        b = tf.get_variable('multi_rnn_cell/cell_1/lstm_cell/bias', shape=multi_rnn_cells[3].shape)
        print(a, "variable")
        a = tf.assign(a, multi_rnn_cells[1])
        b = tf.assign(b, multi_rnn_cells[3])

    reshape_2 = tf.reshape(outputs, [-1, 3, 4, 16])
    conv2_6 = tf.cast(conv2_6, tf.float32)
    dconv2_6_in = tf.concat([reshape_2, conv2_6], 3)
    dconv2_5 = upconv2d_layer(dconv2_6_in, output_shape = (n_feature,n_height,9,16), w_init=get_parameter(model_dir, 'generator_model/upcv6/w')
                              ,strides=strides,padding='VALID',activation_function_type='lrelu', bias=False)

    conv2_5 = tf.cast(conv2_5, tf.float32)
    dconv2_5_in = tf.concat([dconv2_5, conv2_5], 3)
    dconv2_4 = upconv2d_layer(dconv2_5_in, output_shape = (n_feature, n_height, 17, 16),w_init=get_parameter(model_dir, 'generator_model/upcv5/w')
                              , strides=strides, padding='SAME', activation_function_type='lrelu', bias=False)

    conv2_4 = tf.cast(conv2_4, tf.float32)
    dconv2_4_in = tf.concat([dconv2_4, conv2_4], 3)
    dconv2_3 = upconv2d_layer(dconv2_4_in, output_shape = (n_feature, n_height, 33, 8), w_init=get_parameter(model_dir, 'generator_model/upcv4/w')
                              , strides=strides, padding='SAME',activation_function_type='lrelu', bias=False)

    conv2_3 = tf.cast(conv2_3, tf.float32)
    dconv2_3_in = tf.concat([dconv2_3, conv2_3], 3)
    dconv2_2 = upconv2d_layer(dconv2_3_in, output_shape = (n_feature, n_height, 65, 8), w_init=get_parameter(model_dir, 'generator_model/upcv3/w')
                              , strides=strides, padding='SAME',activation_function_type='lrelu', bias=False)

    conv2_2 = tf.cast(conv2_2, tf.float32)
    dconv2_2_in = tf.concat([dconv2_2, conv2_2], 3)
    dconv2_1 = upconv2d_layer(dconv2_2_in, output_shape = (n_feature, n_height, 129, 8), w_init=get_parameter(model_dir, 'generator_model/upcv2/w')
                              , strides=strides, padding='SAME',activation_function_type='lrelu', bias=False)

    conv2_1 = tf.cast(conv2_1, tf.float32)
    dconv2_1_in = tf.concat([dconv2_1, conv2_1], 3)
    dconv2_0 = upconv2d_layer(dconv2_1_in, output_shape = (n_feature, n_height, 257, 1), w_init=get_parameter(model_dir, 'generator_model/upcv1/w')
                              , strides=[1, 1, 2, 1],padding='SAME',activation_function_type='lrelu', bias=False)

    conv2_7 = conv2d_layer(dconv2_0,get_parameter(model_dir, 'generator_model/cv7/w'), get_parameter(model_dir, 'generator_model/cv7/b')
                           ,strides = strides, activation_function_type = 'sigmoid', padding = 'VALID')


    # define plot imag
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    sess.run(a)
    sess.run(b)
    tvars = sess.run([tv, outputs])

    conv2_7, dconv2_0, dconv2_1, dconv2_2, dconv2_3, dconv2_4, dconv2_5, conv2_6, conv2_5, conv2_4, conv2_3, conv2_2, conv2_1 = sess.run(
        [conv2_7, dconv2_0, dconv2_1, dconv2_2, dconv2_3, dconv2_4, dconv2_5, conv2_6, conv2_5, conv2_4, conv2_3, conv2_2, conv2_1])
    plt.figure(1)
    plt.subplot(351);plt.imshow(x[:, 1, :, 0].T);plt.title('original image')
    plt.subplot(352);plt.imshow(conv2_1[:, 1, :, 0].T);plt.title('conv2_1 image')
    plt.subplot(353);plt.imshow(conv2_2[:, 1, :, 0].T);plt.title('conv2_2 image')
    plt.subplot(354);plt.imshow(conv2_3[:, 1, :, 0].T);plt.title('conv2_3 image')
    plt.subplot(355);plt.imshow(conv2_4[:, 1, :, 0].T);plt.title('conv2_4 image')
    plt.subplot(356);plt.imshow(conv2_5[:, 1, :, 0].T);plt.title('conv2_5 image')
    plt.subplot(357);plt.imshow(conv2_6[:, 1, :, 0].T);plt.title('conv2_6 image')

    plt.subplot(358);plt.imshow(dconv2_5[:, 1, :, 0].T);plt.title('dconv2_5 image')
    plt.subplot(359);plt.imshow(dconv2_4[:, 1, :, 0].T);plt.title('dconv2_4 image')
    plt.subplot(3,5,10);plt.imshow(dconv2_3[:, 1, :, 0].T);plt.title('dconv2_3 image')
    plt.subplot(3,5,11);plt.imshow(dconv2_2[:, 1, :, 0].T);plt.title('dconv2_2 image')
    plt.subplot(3,5,12);plt.imshow(dconv2_1[:, 1, :, 0].T);plt.title('dconv2_1 image')
    plt.subplot(3,5,13);plt.imshow(dconv2_0[:, 1, :, 0].T);plt.title('dconv2_0 image')

    plt.subplot(3,5,15);plt.imshow(conv2_7[:, 0, :, 0].T);plt.title('conv2_7 image')
    plt.show()

    # plt.figure(2)
    # out = tf.reshape(conv2_7, [n_feature, 257])
    # return out

def mini_enhanced():
    audio_dir = 'C:/Users/asus/Desktop/dnn_demo/audio/'
    save_dir = 'C:/Users/asus/Desktop/dnn_demo/mini_enhanced/'
    scaler_path = 'C:/Users/asus/Desktop/dnn_demo/mini_scale/scale_512_256_emph.hdf5'
    data_type = '.wav'
    model_dir = 'C:/Users/asus/Desktop/dnn_demo/mini_model/'
    n_frame = 7
    n_window = 512
    n_feature = int(n_window / 2) + 1
    n_overlap = 256
    fs = 16000
    scale = True
    n_forward = 3
    n_backward = 3
    n_frame = n_forward + n_backward + 1

    get_parameter(model_dir, 'generator_model/cv5/w')

    fp = h5py.File(scaler_path, 'r')
    mean = np.array(fp.get('feature_lps_mean'))
    var = np.array(fp.get('feature_lps_var'))
    std = np.sqrt(var)

    fp.close()

    start_frame = 2
    end_frame = 0
    n_frame = start_frame - end_frame + 1

    nat_frame = 6

    win_func = librosa.filters.get_window('hanning', n_window)

    file_name_list = search_file(audio_dir, data_type)

    b, a = signal.butter(2, 200 / fs * 2, 'highpass')
    w, h = signal.freqz(b, a)

    for file_name in file_name_list:

        file_path = os.path.join(file_name[0], file_name[1])
        audio = read_audio(file_path, fs)

        #        audio = pre_emph(audio, coeff=0.97)

        audio_magnitude, audio_angle = calculate_audio_spectrogram(audio, n_window, n_overlap, win_func,
                                                                   'magnitude')

        audio_feature_map = generate_feature_map_asymmetry(audio_magnitude, start_frame, end_frame)

        nframes = audio_angle.shape[0]
        audio_angle = audio_angle[start_frame:nframes - end_frame, :]
        audio_magnitude = audio_magnitude[start_frame:nframes - end_frame, :]

        audio_feature_map = calculate_log(audio_feature_map)

        audio_feature_map = audio_feature_map.reshape((audio_feature_map.shape[0], n_frame * n_feature))

        z = np.mean(audio_feature_map[0:nat_frame, :], axis=0)

        z = np.tile(z, (audio_feature_map.shape[0], 1))

        audio_feature_map = audio_feature_map.reshape((audio_feature_map.shape[0], n_frame, n_feature))
        z = z.reshape((z.shape[0], n_frame, n_feature))

        if scale:
            audio_feature_map = scale_feature(audio_feature_map, mean, std)
            z = scale_feature(z, mean, std)

        audio_feature_map_in = np.zeros((audio_feature_map.shape[0], n_frame, n_feature, 2))

        audio_feature_map_in[:, :, :, 0] = audio_feature_map
        audio_feature_map_in[:, :, :, 1] = z

        model_build(audio_feature_map_in)
        # audio_feature_map_prediction = model.prediction(audio_feature_map_in)
        #
        # audio_feature_map_prediction = audio_feature_map_prediction * audio_magnitude
        #
        # predict_audio = restore_audio(audio_feature_map_prediction, audio_angle, n_window, n_overlap, win_func,
        #                               'magnitude')
        #
        # predict_audio = de_emph(predict_audio, coeff=0.97)
        # predict_audio = signal.filtfilt(b, a, predict_audio)
        #
        # predict_audio = predict_audio[int(n_window / 2):len(predict_audio) - int(n_window / 2)]
        #
        # if (np.max(np.abs(predict_audio)) > 1):
        #     predict_audio = predict_audio / np.max(np.abs(predict_audio))
        #
        # save_path = os.path.join(save_dir, file_name[1])
        # soundfile.write(save_path, predict_audio, fs)
        # print('save file:', save_path)

if __name__ == '__main__':
    mini_enhanced()