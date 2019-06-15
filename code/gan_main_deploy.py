#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import datetime
import time
import functions
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



################################


test_data_path = '/home/dl-linux/Desktop/CGAN_derain_dense/training/40.jpg'
model_path = '/home/dl-linux/Desktop/CGAN_derain_dense/model/model.ckpt'
################################






batch_size = 1
epochs=1000

lambda_a = 3                          #GAN coefficient
lambda_p = 0.015                      #vgg coefficient
lambda_e = 15                         #raw coefficient

discrminator_learning_rate = 0.002
generator_learning_rate = 0.0002
beta1 = 0.5

label_switch_frequency = 10




#set up the learning policy
#global_step = tf.Variable(0, trainable = False)
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1 * (700/5), 0.995, staircase=False)


#set up the placeholder

batch_sizes = tf.placeholder(tf.int32)
derain_placeholder = tf.placeholder(tf.float32, shape=(batch_size,224,224,3))
rain_placeholder = tf.placeholder(tf.float32, shape=(batch_size,224,224,3))


d_watch_placeholder = tf.placeholder(tf.float32)
g_watch_placeholder = tf.placeholder(tf.float32)


img_ground_truth_for_vgg16 = tf.placeholder(tf.float32, shape=(batch_size,224,224,3))


D_Loss_placeholder = tf.placeholder(tf.float32)
G_Loss_placeholder = tf.placeholder(tf.float32)



#Generator
Gz = functions.generator(rain_placeholder,batch_size)

#Discriminator

fake = tf.concat([Gz, rain_placeholder], 3)
real = tf.concat([derain_placeholder, rain_placeholder], 3)

Dg = functions.discriminator(fake)            #for generated image and groumd truth
Dg_truth = functions.discriminator(real)      #for derain image and groumd truth

#VGG19
vgg19_features_output_gen = functions.vgg_19(Gz)
vgg19_features_gt = functions.vgg_19(derain_placeholder)


#Define generator loss

L_E = tf.reduce_mean( 1 * ((tf.abs(tf.math.subtract(derain_placeholder,Gz)))))
L_A = tf.reduce_mean(-1 * tf.math.log(tf.clip_by_value(tf.math.sigmoid(Dg), 1e-10, 1)))
L_P = tf.reduce_mean( 1 * ((tf.abs(tf.math.subtract(vgg19_features_output_gen,vgg19_features_gt)))))

L_RP = lambda_e * L_E + lambda_a*L_A + lambda_p * L_P

g_loss = L_RP


#Define discriminator loss (apply label switch)

d_loss = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg, dtype=tf.float32) ) + tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg_truth, labels = tf.ones_like(Dg_truth, dtype=tf.float32) )))

d_loss_flip = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg, dtype=tf.float32) ) + tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg_truth, labels = tf.zeros_like(Dg_truth, dtype=tf.float32) )))




# Trainable parameters


tvars = tf.global_variables()


gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

d_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]




# Draw the tensorboard

sess = tf.Session()


tf.summary.scalar('d_weight_watch', d_watch_placeholder)
tf.summary.scalar('g_weight_watch', g_watch_placeholder)

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss', d_loss)

tf.summary.scalar('Raw_loss', L_E)
tf.summary.scalar('VGG_loss', L_P)
tf.summary.scalar('GAN_loss', L_A)

output_img = tf.summary.image('Output', Gz, max_outputs = 1)                     #record the input image
target_img = tf.summary.image('Target', derain_placeholder, max_outputs = 1)     #record the generator output image
input_img = tf.summary.image('Input', rain_placeholder, max_outputs = 1)         #record the ground truth image


tf.summary.merge([output_img, target_img, input_img])


merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


#print out trainable parameters
print d_vars
print gen_vars


#define optimizer

d_trainer = tf.train.GradientDescentOptimizer(discrminator_learning_rate)             # discriminator use SGD

gradients_of_discriminator = d_trainer.compute_gradients(d_loss, d_vars)
d_train = d_trainer.apply_gradients(gradients_of_discriminator)

gradients_of_discriminator_flip = d_trainer.compute_gradients(d_loss_flip, d_vars)
d_train_flip = d_trainer.apply_gradients(gradients_of_discriminator_flip)


g_trainer = tf.train.AdamOptimizer(generator_learning_rate,beta1 = beta1)                           # generator use Adam
gradients_of_generator = g_trainer.compute_gradients(g_loss, gen_vars)
g_train = g_trainer.apply_gradients(gradients_of_generator)


saver = tf.train.Saver()


image = functions.load_img(test_data_path)



with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  saver.restore(sess, model_path)


  imgs = sess.run(image)



  go = imgs[:,224:,:]
  plt.figure(1)
  plt.imshow(go)



  go = np.expand_dims(go, axis = 0)  
  print go.shape
  print go
  result = sess.run(Gz, {rain_placeholder: go})
  print result.shape
  plt.figure(2)
  plt.imshow(result[0,:,:,:])


  plt.show()

