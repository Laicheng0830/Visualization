"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/11/25 14:24
"""
import soundfile
import librosa
import numpy as np

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio


def extract_frames(data, nsamples, frame_len, overlap):
    end_pos = nsamples - frame_len
    if (end_pos < 0):
        return -1

    move_len = frame_len - overlap
    pos = 0
    nframes = int(end_pos / move_len) + 1
    frame_matrix = np.zeros((nframes, frame_len))

    i = 0
    while pos <= (end_pos):
        frame = data[pos:pos + frame_len]
        frame_matrix[i, :] = frame
        i = i + 1
        pos = pos + move_len

    return frame_matrix


def feature_transform(data, mode, win):
    nsamples = data.shape[0]
    fft_len = int(data.shape[1] / 2) + 1

    data_feature = np.zeros((nsamples, fft_len))
    data_feature = data_feature.astype(np.complex64)
    nframes = np.shape(data)[0]

    for i in range(nframes):
        fft_data = (np.fft.fft(data[i, :] * win))
        data_feature[i, :] = fft_data[0:fft_len]

    if (mode == 'magnitude'):
        data_feature_magnitude = np.abs(data_feature)
        data_feature_angle = np.angle(data_feature)
    #        data_feature_list = np.hstack((data_feature_list_magnitude,data_feature_list_angle))

    if (mode == 'complex'):
        data_feature_magnitude = np.real(data_feature)
        data_feature_angle = np.imag(data_feature)
    #        data_feature_list = np.hstack((data_feature_list_real,data_feature_list_imag))

    return data_feature_magnitude, data_feature_angle

def calculate_audio_spectrogram(data, n_frame, n_overlap, win, mode, pad=True):

    if (pad == True):
        data = np.pad(data, int(n_frame / 2), mode='reflect')

    nsamples = len(data)

    frame_matrix = extract_frames(data, nsamples, n_frame, n_overlap)

    data_feature_magnitude, data_feature_angle = feature_transform(frame_matrix, mode, win)

    return data_feature_magnitude, data_feature_angle

def generate_feature_map_asymmetry(x, n_forward, n_backward):
    len_x, n_in = x.shape
    n_frame = n_forward + n_backward + 1
    n_sample = len_x - n_frame + 1
    x3d = np.zeros((n_sample, n_frame, n_in))
    for i in range(n_sample):
        x3d[i, :, :] = x[i:i + n_frame, :]
    return x3d

def calculate_log(x,mini_scalar=1e-08):
    return np.log(x + mini_scalar)

