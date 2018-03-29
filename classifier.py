import numpy as np
import matplotlib.pyplot as pyplot
import h5py
import scipy
from PIL import Image
from scipy import ndimage

# extras for debugging
import math

# dataset loader
import utils

# dataset parameters
training_path = './training_data'			# path of training data containing class sub-directories (image files)
image_size = 128 							# length and width to uniformly format training data
classes = ['apple', 'orange', 'banana'] 	# classes of images to classify
c_len = len(classes)						# number of classes to be used for training
validation_size = 0.2 						# randomly chosen 20% of training data to be used as validation data

# model parameters
iteration_count = 1000						# number of times to apply gradient descent
learning_rate = 0.005						# size of gradient step
show_cost = True 							# show cost every 100 iterations

# loading data_set object
data_set = utils.read_data_sets(training_path, image_size, classes, validation_size)

# designating training objects
training_images = data_set.train.images 		# image np.array w/ shape: (image_size, image_size, channel_depth)
training_labels = data_set.train.labels 		# class label array (exempli gratia '[1.0, 0, 0]' from apple)
training_class_set = data_set.train.class_set 	# class label string array (e.g. 'apple')
training_file_name = data_set.train.image_names

# designating validation objects
validation_images = data_set.valid.images
validation_labels = data_set.valid.labels
validation_class_set = data_set.valid.class_set
validation_file_name = data_set.valid.image_names

# reshaping using matrix transposition
training_images = training_images.reshape(training_images.shape[0], -1).T
validation_images = validation_images.reshape(validation_images.shape[0], -1).T
training_labels = training_labels.T
validation_labels = validation_labels.T

#data is now properly formatted and defined respectively

"""
flattening pixels to single layer using transpose function of image pixel matrix
shape: (image_size * image_size * channel_dept, len(training_images))
"""
#flattened = training_images.reshape(training_images.shape[0], -1).T


def sigmoid(z):
	"""
	Computing the sigmoid of z

	Parameters: 
		-- z = w^T * x^i + b 
		-- w^T: specific weight associated with nueron index from previous layer
		-- x^i: specific nueron value from previous layer
		-- b: bias associated with neuron 

	Return:
	s: result of applying sigmoid activation function (domain in R, returns monotonicingly increasing value between 0 and 1)
		s = 1 / (1 + e^-z)
	"""

	s = 1 / (1 + np.exp(-z)) #definition of the sigmoid function
	return s

def init_zero(dimension):
	"""
	Parameters:
		-- dimension: the length of matrix to be initialized

	Initializes:
		-- w (weight array): zero array w/ shape: (image_size * image_size * channel_depth, 1)
		-- b (bias value): as zero 
	"""
	w = np.zeros(shape=(dimension, 3))
	b = 0

	# shape and type check
	assert(w.shape == (dimension, 3)), "w in not in proper form: init_zero(dimension)"
	assert(isinstance(b, float) or isinstance(b, int)), "b is not of type int/float"

	return w, b

def cross_entropy_cost(m, A, L):
	"""
	Cross-Entropy Cost function of logistic-regression.

	Parameters:
		-- m: count of items in set
		-- A: numpy array of activation values corresponding to each neuron w/ shape: (1, m)
		-- L: true label array to identify true class type w/ shape: (1, m)

	Return:
		-- cost: negative log-probability cost for logistic regression


	Notes:
		-- Cross-Entropy Cost is calculated in a logarithmic fashion as prediction function (sigmoid) is non-linear.
		-- 'Squaring this prediction as we do in MSE results in a non-convex function with many local minimums. 
			If our cost function has many local minimums, gradient descent may not find the optimal global minimum.'
		-- Cross-Entropy Cost penalizes confident wrong predictions more than rewards confident correct predictions.
	

	Calculation of Cross-Entropy Cost:
		C = (-1/m) * Sigma([y^i * log(A^i] + [1 - L^i] * [log(1 - A^i])) 
			from i = 1 to m
	"""

	#Note: Using numpy masked array np.ma for values of log(0)
	cost = (-1 / m) * np.sum(L * np.log(A) + (1 - L) * (np.ma.log(1 - A)))
	if math.isnan(cost):
		print(np.log(A))
		print(1-A)
		print(np.log(1-A))
		exit()

	# Sanity checks
	cost = np.squeeze(cost) 	#squeeze() removes single dimensional elements from the array: e.g. (1, 3, 1) -> (3,)
	assert(cost.shape == ()) 	#checks if cost value is a scalar

	return cost

def propagate(w, b, image_matrix, true_labels):
	"""
	Forwards and Backwards Propagation of Error


	Parameters: 
		-- w: weights numpy array w/ shape: (image_size * image_size * channel_depth, 1)
		-- b: specific bias, scalar value
		-- image_matrix: flattened image matrix w/ shape (image_size * image_size * channel_depth, image_matrix.shape[1])
		-- true_labels: correct "label" array for each image w/ shape (1, image_matrix.shape[1])

	Returns:
		-- gradients:
		-- cost: 

	"""

	m = image_matrix.shape[1] # image count

	"""
	FORWARD PROPAGATION: output compared to actual to obtain cost (error)
	activation_val: 
		sigmoid(z) w/ z = w^T * x^i + b
	cost: cross_entropy_cost(m, A, L)
	"""

	# debugging
	#flattened = training_images.reshape(training_images.shape[0], -1).T
	#print("flatX: %s\n w.T: %s\n" % (flattened.shape, w.T.shape))



	activation_val = sigmoid(np.dot(w.T, image_matrix) + b) 
	cost = cross_entropy_cost(m, activation_val, true_labels)

	#cost = (-1 / m) * np.sum(true_labels * np.log(activation_val) + (1 - true_labels) * (np.log(1 - activation_val)))

	"""
	BACKWARD PROPAGATION: to obtain gradient of loss for weights and biases as to minimize error of network
	dw: gradient of loss with respect to w
	db: gradient of loss with respect to b
	"""
	dw = (1 / m) * np.dot(image_matrix, (activation_val - true_labels).T)
	db = (1 / m) * np.sum(activation_val - true_labels)

	assert(dw.shape == w.shape) #checks if weight gradient retains weight matrix shape
	assert(db.dtype == float)	#checks if bias gradient is a scalar

	# format into single object for return
	gradients = { 
		"dw": dw,
		"db": db
	}

	return gradients, cost

def gradient_descent(w, b, image_matrix, true_labels, iteration_count, learning_rate, show_cost):
	"""
	Gradient Descent optimization of weights and biases

	Parameters:
		-- w: weights array w/ shape: (image_size * image_size * channel_depth, 1)
		-- b: bias scalar
		-- image_matrix: flattened image matrix w/ shape (image_size * image_size * channel_depth, m)
		-- true_labels: correct "label" array for each image w/ shape (1, m)
		-- interation_count: the number of iterations that the function will loop through during optimization
		-- learning_rate: 
		-- show_cost: print cost value to console every 100 iterations

	Return:
		-- 

	Notes:
		-- 
	"""

	costs = []

	for i in range(iteration_count):
		gradients, cost = propagate(w, b, image_matrix, true_labels)
		# if math.isnan(cost):
		# 	A = sigmoid(np.dot(w.T, image_matrix) + b)
		# 	print(np.squeeze(A))
		# 	print(cross_entropy_cost(image_matrix.shape[1], A, true_labels))

		dw = gradients['dw']
		db = gradients['db']

		w = w - learning_rate * dw
		b = b - learning_rate * db

		if i % 100 == 0:
			costs.append(cost)

		if show_cost and i % 100 == 0 and i != 0:
			print('Iteration: %i, Cost: %f' % (i, cost))

	parameters = {
	"w": w,
	"b": b
	}

	gradients = {
	"dw": dw,
	"db": db,
	}

	return parameters, gradients, costs

def predict(w, b, image_matrix):
	"""
	Makes a prediction of label using parameters obtained from learning

	Parameters:
		-- w: weights array w/ shape: (image_size * image_size * channel_depth, 1)
		-- b: bias scalar
		-- image_matrix: flattened image matrix w/ shape (image_size * image_size * channel_depth, m)

	Returns:
		-- prediction_labels: numpy array containing all predictions for data in image_matrix

	Notes:

	"""
	m = image_matrix.shape[1] 					# grab set size again
	prediction_labels = np.zeros((3, m))		# init vector

	activation_val = sigmoid(np.dot(w.T, image_matrix) + b) 

	for i in range(activation_val.shape[1]):
		for j in range(3):
			if activation_val[j, i] > 0.5:
				prediction_labels[j, i] = 1
			else:
				prediction_labels[j, i] = 0

	assert(prediction_labels.shape == (3, m))

	return prediction_labels

def model(training_images, training_labels, validation_images, validation_labels, iteration_count, learning_rate, show_cost):
	"""
	"""

	# init weight and bias arrays
	w, b = init_zero(training_images.shape[0]) 

	# train model and obtain weight and bias 
	parameters, gradients, costs = gradient_descent(w=w, b=b, image_matrix=training_images, true_labels=training_labels,
	 iteration_count=iteration_count, learning_rate=learning_rate, show_cost=show_cost)

	w = parameters["w"]
	b = parameters["b"]

	prediction_training_labels = predict(w, b, training_images)
	prediction_validation_labels = predict(w, b, validation_images)

	training_accuracy = (1 - np.mean(np.abs(prediction_training_labels - training_labels)))
	validation_accuracy = (1 - np.mean(np.abs(prediction_validation_labels - validation_labels)))

	print("training accuracy: %s" % str(training_accuracy))
	print("validation accuracy: %s" % str(validation_accuracy))

	data = {
	"costs": costs,
	"prediction_training_labels": prediction_training_labels,
	"prediction_validation_labels": prediction_validation_labels,
	"original_training_labels": training_class_set,
	"original_validation_labels": validation_class_set,
	"w": w,
	"b": b,
	"learning_rate": learning_rate,
	"interation_count": iteration_count
	}

	return data

def train():
	data = model(training_images=training_images, training_labels=training_labels,
	 validation_images=validation_images, validation_labels=validation_labels,
	  iteration_count=iteration_count, learning_rate=learning_rate, show_cost=show_cost)

	return data

















