# code based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import cv2
import random
from load_data import LoadData as GetCIFARData

FILE_NAME = 'CIFAR_HIDDEN2_NN'
cifar = GetCIFARData(one_hot=True)

# Parameters
n_input = 32*32*3 # my CIFAR data input (img shape = 32*32)
n_classes = 5 # my CIFAR total classes (0-5)
starter_learning_rate = .0001
epochs = 2000
batch_size = 100
display_step = 200

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

def InitWeightMaker(init):
	cache = {'init': init}
	def MakeWeight(out):
		inp = cache['init']
		cache['init'] = out
		return tf.Variable( tf.random_normal([inp, out]) )
	return MakeWeight

for factor1 in [2, 4, 8, 16]:
	for factor2 in [2, 4, 8, 16]:
		if factor1 >= factor2:
			continue

		# Network Paramters
		hidden1 = n_input/factor1 # number of features in 1st hidden layer
		hidden2 = n_input/factor2 # number of features in 2nd hidden layer
		print ("Now processing with h1=", hidden1, "h2=", hidden2)

		###############
		# BUILD MODEL #
		###############
		with tf.device('/gpu:0'):

			# tf Graph input
			x = tf.placeholder("float", shape=[None, n_input])
			labels = tf.placeholder("float", shape=[None, n_classes])

			MakeWeight = InitWeightMaker(n_input)
			MakeBias = lambda x : tf.Variable( tf.random_normal([x]) )
			MakeWeightBias = lambda x: ( MakeWeight(x), MakeBias(x) )
			WeightedOutput = lambda i, w, b: tf.add( tf.matmul(i, w), b )
			Activate = lambda x: tf.nn.relu(x)

			W1, b1 = MakeWeightBias(hidden1)
			layer1 = Activate( WeightedOutput(x, W1, b1) )

			W2, b2 = MakeWeightBias(hidden2)
			layer2 = Activate( WeightedOutput(layer1, W2, b2) )

			W3, b3 = MakeWeightBias(n_classes)
			logits = WeightedOutput(layer2, W3, b3)
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

			# Define loss and optimizer
			global_step = tf.placeholder(tf.int32, [])
			learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, epochs/2, .5)
			backprop = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

			##############################
			####### EVALUATE MODEL #######
			##############################

			# Launch the graph
			errors = []
			myaccuracy = None
			with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
				# Initializing the variables
				init = tf.global_variables_initializer()
				sess.run(init)
				
				# Training cycle
				for epoch in range( epochs ):
					avg_cost = 0
					total_batch = int( cifar.train.num_examples/batch_size )

					# Loop over all batches
					for i in range(total_batch):
						batch_x, batch_y = cifar.train.next_batch(batch_size)
						_, c = sess.run([backprop, cost], 
										feed_dict={x:batch_x, labels:batch_y, global_step:epoch})
						avg_cost += c/total_batch
					
					# Display logs per epoch step
					errors.append(avg_cost)
					if epoch % display_step == 0:
						print("Epoch: ", '%4d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
				print("Optimization Finished!")

				correct_prediction = tf.equal( tf.argmax(logits, 1), tf.argmax(labels, 1) )
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				myaccuracy = accuracy.eval({x: cifar.test.images, labels: cifar.test.labels})
				print("Accuracy: ", myaccuracy)
			
			########
			# SAVE #
			########
			name = FILE_NAME + "-h1=" + repr(hidden1) + "-h2=" + repr(hidden2)
			title = "Two hidden layers h1 and h2 where h1=" + repr(hidden1) + " h2=" + repr(hidden2) 
			SaveAccuracy(name, errors, myaccuracy)
			SaveErrorPlot(name, title, errors)
			print ("DONE with " + name)

print ("DONE with processing " + FILE_NAME)