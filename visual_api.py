"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/11/19 14:19
"""
import tensorflow as tf
import numpy as np

def conv2d_layer(data, kenerl_w, biases, strides=[1, 1, 1, 1], padding='SAME',
                 activation_function_type='lrelu', keep_prob=1,
                 bias=True, dropout=False):
    cov = tf.nn.conv2d(data, kenerl_w, strides=strides, padding=padding)
    if (bias == True):
        h = activation_function(cov + biases, activation_function_type)
    else:
        h = activation_function(cov, activation_function_type)
    if (dropout == True):
        out = tf.nn.dropout(h, keep_prob)
    else:
        out = h
    return out


def upconv2d_layer(data, output_shape, w_init,b_init = 0, strides=[1, 1, 1, 1], padding='SAME',
                   activation_function_type='lrelu',keep_prob=1,bias=False):
    conv = tf.nn.conv2d_transpose(data, w_init, output_shape, strides, padding=padding)
    if (bias == True):
        h = activation_function(conv + b_init, activation_function_type)
    else:
        h = activation_function(conv, activation_function_type)

    if ((keep_prob < 1) and keep_prob > 0):
        out = tf.nn.dropout(h, keep_prob)
    else:
        out = h
    return out


def leaky_relu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        relu = f1 * x + f2 * abs(x)
        return relu

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def activation_function(x,activation_function_type):
    if(activation_function_type=='lrelu'):
        h = leaky_relu(x)
    if(activation_function_type=='tanh'):
        h = tf.tanh(x)
    if(activation_function_type=='sigmoid'):
        h = tf.sigmoid(x)
    if(activation_function_type=='relu'):
        h = tf.nn.relu(x)
    if(activation_function_type=='linear'):
        h = x
    if(activation_function_type=='softmax'):
        h = tf.nn.softmax(x)
    return h

def lstm_hc1(x, xc, h, hc, c, cc, w1icfo, b1icfo):
    icfo = (np.concatenate((x, h))).dot(w1icfo) + b1icfo
    it = sigmoid(icfo[0*hc:1*hc])
    ct = np.tanh(icfo[1*hc:2*hc])
    ft = sigmoid(icfo[2*hc:3*hc])
    ot = sigmoid(icfo[3*hc:4*hc])
    c1 = ft*c + it*ct
    h1 = ot*np.tanh(c1)
    return h1, c1

def lstm_hc2(x, xc, h, hc, c, cc, w2icfo, b2icfo):
    icfo = (np.concatenate((x, h))).dot(w2icfo) + b2icfo
    it = sigmoid(icfo[0*hc:1*hc])
    ct = np.tanh(icfo[1*hc:2*hc])
    ft = sigmoid(icfo[2*hc:3*hc])
    ot = sigmoid(icfo[3*hc:4*hc])
    c2 = ft*c + it*ct
    h2 = ot*np.tanh(c2)
    return h2, c2

def lstm_layer(x, weight, biase, weight2, biase2):
    xc = 64
    hc = 64
    cc = 64
    y=[]
    hc = 64
    cc = 64

    hi1 = np.zeros((hc,), dtype='float')
    ci1 = np.zeros((cc,), dtype='float')
    hi2 = np.zeros((hc,), dtype='float')
    ci2 = np.zeros((cc,), dtype='float')

    for i in range(len(x)):
        xi1 = x[i,:]
        ho1, co1 = lstm_hc1(xi1, xc,hi1, hc,ci1, cc, weight, biase)
        hi1 = ho1
        ci1 = co1
        xi2 = ho1
        ho2, co2 = lstm_hc2(xi2, xc,hi2, hc,ci2, cc, weight2, biase2)
        hi2 = ho2
        ci2 = co2
        y.append(ho2)
    y = np.array(y)
    return y, hi1, ci1, hi2, ci2