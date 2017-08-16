from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import cv2
from load_data import LoadData as GetCIFARData
from tensorflow.examples.tutorials.mnist import input_data

FILE_NAME = 'CIFAR_SIMPLE_NN'
dataset = GetCIFARData(img_vectorized = True, one_hot=True)

n_input = len( dataset.train.images[0] )
n_classes = 5
starter_learning_rate = .05
epochs = 2000
batch_size = 100
display_step = 1

def SaveAccuracy(name, errors, a):
	import simplejson
	f = open( os.path.join("accuracies", name), "w" )
	f.write("%2.4f" % a)
	f.write("\n\n")
	simplejson.dump(errors, f)
	f.close()

def SaveErrorPlot(name, title, errors):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(errors)

	plt.suptitle(title, fontsize=20)
	plt.xlabel('epoch')
	plt.ylabel('average error')
	
	plt.savefig( os.path.join("plots", name) + ".png")

with tf.device('/gpu:0'):
	# Build the model: 1 layer perceptron
	x = tf.placeholder(tf.float32, shape=[None, n_input])
	labels = tf.placeholder(tf.float32, [None, n_classes])

	W = tf.Variable(tf.zeros([n_input, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))
	logits = tf.matmul(x, W) + b
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits) )

	# setup gradient descent with decaying learning rate
	global_step = tf.placeholder(tf.int32, [])
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, epochs/2, .5)
	backprop = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	errors = []
	myaccuracy = None
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		total_batch = int( dataset.train.num_examples/batch_size )
		total_batch += int( bool(dataset.train.num_examples % batch_size) )

		# Train
		for epoch in range(epochs):
			avg_cost = 0

			# Loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = dataset.train.next_batch(batch_size)
				_, c = sess.run([backprop, cost], feed_dict={x: batch_xs, labels:batch_ys, global_step: epoch})
				avg_cost += c/len(batch_xs)

			# Print progress
			if epoch % display_step == 0:
				print("Epoch: ", '%4d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
			errors.append(avg_cost)

		# Test and evaluate
		correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		myaccuracy = sess.run(accuracy, feed_dict={x: dataset.test.images, labels: dataset.test.labels})
		print("Accuracy:", myaccuracy)

	# Plot the errors
	title = "Simple neural network"
	name = FILE_NAME
	SaveAccuracy(name, errors, myaccuracy)
	SaveErrorPlot(name, title, errors)
	print ("DONE with " + FILE_NAME)
