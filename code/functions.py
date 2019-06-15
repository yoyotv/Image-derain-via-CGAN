#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import time
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import functions
import vgg19

def generator(img, batch_size):

  K = 64
  K_2 = 32
  channel_num = 3
  img_h = 224
  img_w = 224
  
  with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

    #get encoder variables
    #e1_w = tf.get_variable('e1_w', [3, 3, channel_num, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    #e1_w = tf.get_variable('gen_v_e1_w', [3, 3, channel_num, K], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    #e1_w = tf.get_variable('gen_v_e1_w', [3, 3, channel_num, K], dtype=tf.float32, initializer=tf.zeros_initializer)
    e1_w = tf.get_variable('gen_v_e1_w', [3, 3, channel_num, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    e1_b = tf.get_variable('gen_v_e1_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    e2_w = tf.get_variable('gen_v_e2_w', [3, 3, K, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    e2_b = tf.get_variable('gen_v_e2_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    e3_w = tf.get_variable('gen_v_e3_w', [3, 3, K, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    e3_b = tf.get_variable('gen_v_e3_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    e4_w = tf.get_variable('gen_v_e4_w', [3, 3, K, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    e4_b = tf.get_variable('gen_v_e4_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    e5_w = tf.get_variable('gen_v_e5_w', [3, 3, K, K_2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    e5_b = tf.get_variable('gen_v_e5_b', [K_2], initializer=tf.truncated_normal_initializer(stddev=0.02))

    e6_w = tf.get_variable('gen_v_e6_w', [3, 3, K_2, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    e6_b = tf.get_variable('gen_v_e6_b', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))

    #get decoder variables
    d1_w = tf.get_variable('gen_v_d1_w', [3, 3, K_2, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    d1_b = tf.get_variable('gen_v_d1_b', [K_2], initializer=tf.truncated_normal_initializer(stddev=0.02))

    d2_w = tf.get_variable('gen_v_d2_w', [3, 3, K, K_2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    d2_b = tf.get_variable('gen_v_d2_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    d3_w = tf.get_variable('gen_v_d3_w', [3, 3, K, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    d3_b = tf.get_variable('gen_v_d3_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    d4_w = tf.get_variable('gen_v_d4_w', [3, 3, K, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    d4_b = tf.get_variable('gen_v_d4_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    d5_w = tf.get_variable('gen_v_d5_w', [3, 3, K, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    d5_b = tf.get_variable('gen_v_d5_b', [K], initializer=tf.truncated_normal_initializer(stddev=0.02))

    d6_w = tf.get_variable('gen_v_d6_w', [3, 3, channel_num, K], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    d6_b = tf.get_variable('gen_v_d6_b', [channel_num], initializer=tf.truncated_normal_initializer(stddev=0.02))

    # encoder




    e1_c = tf.nn.conv2d(img, e1_w, strides=[1, 1, 1, 1], padding='SAME')
    e1_c = e1_c + e1_b
    e1_bn = tf.layers.batch_normalization(e1_c)
    e1_r = tf.nn.leaky_relu(e1_bn)

    e2_c = tf.nn.conv2d(e1_r, e2_w, strides=[1, 1, 1, 1], padding='SAME')
    e2_c = e2_c + e2_b
    e2_bn = tf.layers.batch_normalization(e2_c)
    e2_r = tf.nn.leaky_relu(e2_bn)

    e3_c = tf.nn.conv2d(e2_r, e3_w, strides=[1, 1, 1, 1], padding='SAME')
    e3_c = e3_c + e3_b
    e3_bn = tf.layers.batch_normalization(e3_c)
    e3_r = tf.nn.leaky_relu(e3_bn)

    e4_c = tf.nn.conv2d(e3_r, e4_w, strides=[1, 1, 1, 1], padding='SAME')
    e4_c = e4_c + e4_b
    e4_bn = tf.layers.batch_normalization(e4_c)
    e4_r = tf.nn.leaky_relu(e4_bn)

    e5_c = tf.nn.conv2d(e4_r, e5_w, strides=[1, 1, 1, 1], padding='SAME')
    e5_c = e5_c + e5_b
    e5_bn = tf.layers.batch_normalization(e5_c)
    e5_r = tf.nn.leaky_relu(e5_bn)

    e6_c = tf.nn.conv2d(e5_r, e6_w, strides=[1, 1, 1, 1], padding='SAME')
    e6_c = e6_c + e6_b
    e6_bn = tf.layers.batch_normalization(e6_c)
    e6_r = tf.nn.leaky_relu(e6_bn)
    
    #e6_r = tf.reshape(e6_r, [batch_size,28,28,channel_num])

    # decoder
    d1_dc = tf.nn.conv2d_transpose(e6_r, d1_w, output_shape=[batch_size, img_h, img_w, K_2],strides=[1, 1, 1, 1], padding='SAME')
    d1_dc = d1_dc + d1_b
    d1_bn = tf.layers.batch_normalization(d1_dc)
    d1_r = tf.nn.relu(d1_bn)

    d2_dc = tf.nn.conv2d_transpose(d1_r, d2_w, output_shape=[batch_size, img_h, img_w, K],strides=[1, 1, 1, 1], padding='SAME')
    d2_dc = d2_dc + d2_b
    d2_bn = tf.layers.batch_normalization(d2_dc)
    d2_r = tf.nn.relu(d2_bn)

    d2_concat = tf.add(d2_r, e4_r)
    #d2_concat = d2_r

    d3_dc = tf.nn.conv2d_transpose(d2_concat, d3_w, output_shape=[batch_size, img_h, img_w, K],strides=[1, 1, 1, 1], padding='SAME')
    d3_dc = d3_dc + d3_b
    d3_bn = tf.layers.batch_normalization(d3_dc)
    d3_r = tf.nn.relu(d3_bn)

    d4_dc = tf.nn.conv2d_transpose(d3_r, d4_w, output_shape=[batch_size, img_h, img_w, K],strides=[1, 1, 1, 1], padding='SAME')
    d4_dc = d4_dc + d4_b
    d4_bn = tf.layers.batch_normalization(d4_dc)
    d4_r = tf.nn.relu(d4_bn)

    d4_concat = tf.add(d4_r, e2_r)
    #d4_concat = d4_r

    d5_dc = tf.nn.conv2d_transpose(d4_concat, d5_w, output_shape=[batch_size, img_h, img_w, K],strides=[1, 1, 1, 1], padding='SAME')
    d5_dc = d5_dc + d5_b
    d5_bn = tf.layers.batch_normalization(d5_dc)
    d5_r = tf.nn.relu(d5_bn)

    d6_dc = tf.nn.conv2d_transpose(d5_r, d6_w, output_shape=[batch_size, img_h, img_w, channel_num],strides=[1, 1, 1, 1], padding='SAME')
    d6_dc = d6_dc + d6_b
    d6_bn = tf.layers.batch_normalization(d6_dc)
    d6_r = tf.nn.relu(d6_bn)

    d6_concat = tf.add(d6_r, img)
    #d6_concat = d6_r

    output = tf.math.tanh(d6_concat)     # Dimensions of output : batch_size x img.height x img.width x 3(RGB)

    return output


def discriminator(images, reuse_variables=None):

  kernels_size = 4
  kernels_number = 48

  with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:

    d_w1 = tf.get_variable('d_w1', [kernels_size, kernels_size ,6, kernels_number], initializer=tf.contrib.layers.xavier_initializer())
    d_b1 = tf.get_variable('d_b1',[kernels_number], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 2, 2, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.layers.batch_normalization(d1, name = 'd_batchnorm_1')


    d_w2 = tf.get_variable('d_w2', [kernels_size, kernels_size ,kernels_number, kernels_number*2], initializer=tf.contrib.layers.xavier_initializer())
    d_b2 = tf.get_variable('d_b2',[kernels_number*2], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 2, 2, 1], padding='SAME')
    d2 = d2 +d_b2
    d2 = tf.layers.batch_normalization(d2, name = 'd_batchnorm_2')
    d2 = tf.nn.leaky_relu(d2)


    d_w3 = tf.get_variable('d_w3', [kernels_size, kernels_size ,kernels_number*2, kernels_number*4], initializer=tf.contrib.layers.xavier_initializer())
    d_b3 = tf.get_variable('d_b3',[kernels_number*4], initializer=tf.constant_initializer(0))
    d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 2, 2, 1], padding='SAME')
    d3 = d3 +d_b3
    d3 = tf.layers.batch_normalization(d3,  name = 'd_batchnorm_3')
    d3 = tf.nn.leaky_relu(d3)

    d_w4 = tf.get_variable('d_w4', [kernels_size, kernels_size ,kernels_number*4, kernels_number*8], initializer=tf.contrib.layers.xavier_initializer())
    d_b4 = tf.get_variable('d_b4',[kernels_number*8], initializer=tf.constant_initializer(0))
    d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 1, 1, 1], padding='SAME')
    d4 = d4 +d_b4
    d4 = tf.layers.batch_normalization(d4, name = 'd_batchnorm_4')
    d4 = tf.nn.leaky_relu(d4)

    d_w5 = tf.get_variable('d_w5', [kernels_size, kernels_size ,kernels_number*8 , 1], initializer=tf.contrib.layers.xavier_initializer())
    d_b5 = tf.get_variable('d_b5',[1], initializer=tf.constant_initializer(0))
    d5 = tf.nn.conv2d(input=d4, filter=d_w5, strides=[1, 1, 1, 1], padding='SAME')
    d5 = d5 +d_b5


    d5 = tf.reshape(d5, [-1, 28*28*1])

    d5 = tf.nn.dropout(d5, keep_prob = 0.5)

    output = tf.layers.dense(d5, 1, name = 'd_fc')
    #output = tf.nn.tanh(output)

    return output

def load_img(path):
  img_string = tf.read_file(path)
  img = tf.image.decode_jpeg(img_string, channels=3)
  #img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize_images(img, [224, 448])
  img /= 255.0
  return img

def load_img2(path):
  img_string = tf.read_file(path)
  img = tf.image.decode_jpeg(img_string, channels=3)
  #img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize_images(img, [224, 224])
  img /= 255.0
  return img

import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg

import numpy as np
import functions
import time
import cv2
import matplotlib.pyplot as plt


def vgg_19(img):

  model = vgg19.Vgg19()
  #img = tf.image.resize_images(img, [224, 224])
  layer = model.feature_map(img)
  return layer

