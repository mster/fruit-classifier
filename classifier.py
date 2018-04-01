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
original_training_images = data_set.train.images 		# image np.array w/ shape: (image_size, image_size, channel_depth)
original_training_labels = data_set.train.labels 		# class label array (exempli gratia '[1.0, 0, 0]' from apple)
training_class_set = data_set.train.class_set 			# class label string array (e.g. 'apple')
training_file_name = data_set.train.image_names 		# original unique image file names

# designating validation objects
original_validation_images = data_set.valid.images
original_validation_labels = data_set.valid.labels
validation_class_set = data_set.valid.class_set
validation_file_name = data_set.valid.image_names


"""
Reshaping data arrays using matrix transposition
flattening color pixels to single array using transpose function of image pixel matrix
	*_images shape: (image_size * image_size * channel_depth, data_set_size)
	*_labels shape: (data_set_size, channel_depth)
"""
training_images = original_training_images.reshape(original_training_images.shape[0], -1).T
validation_images = original_validation_images.reshape(original_validation_images.shape[0], -1).T
training_labels = original_training_labels.T
validation_labels = original_validation_labels.T 
# data is now properly formatted and defined respectively


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
		C = (-1 / m) * Sigma([L[i] * log(A[i]) + (1 - L[i]) * (log(1 - A[i])) 
			from i = 1 to m
	"""
	cost = (-1 / m) * np.sum(L * np.log(A) + (1 - L) * (np.ma.log(1 - A))) #Note: Using numpy masked array np.ma for values of log(0)


	# Sanity checks
	cost = np.squeeze(cost) 	#squeeze() removes single dimensional elements from the array: e.g. (1, 3, 1) -> (3,)
	assert(cost.shape == ()) 	#checks if cost value is a scalar

	return cost

def propagate(w, b, image_matrix, true_labels):
	"""
	Forwards and Backwards Propagation of Error.

	Parameters: 
		-- w: weights numpy array w/ shape: (image_size * image_size * channel_depth, 1)
		-- b: specific bias, scalar value
		-- image_matrix: flattened image matrix w/ shape (image_size * image_size * channel_depth, image_matrix.shape[1])
		-- true_labels: correct "label" array for each image w/ shape (1, image_matrix.shape[1])

	Returns:
		-- gradients: the weight and bias gradients computed from the activation layer
		-- cost: the cross entropy cost of the logistic regression 

	"""

	m = image_matrix.shape[1] # image count

	"""
	FORWARD PROPAGATION: output compared to actual to obtain cost (error)
		-- activation_layer: sigmoid of the linear function
			sigmoid(z) w/ z = w^T * x^i + b
		-- cost: see cross_entropy_cost(m, A, L)
	"""
	activation_layer = sigmoid(np.dot(w.T, image_matrix) + b) 
	cost = cross_entropy_cost(m, activation_layer, true_labels)

	"""
	BACKWARD PROPAGATION: to obtain gradient of loss for weights and bias as to minimize error of network
		-- dw: gradient of loss with respect to w
		-- db: gradient of loss with respect to b
	"""
	dw = (1 / m) * np.dot(image_matrix, (activation_layer - true_labels).T) 
	db = (1 / m) * np.sum(activation_layer - true_labels)

	# sanity check
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
	Gradient Descent optimization of weights and bias scaled by learning rate parameter

	Parameters:
		-- w: weights array w/ shape: (image_size * image_size * channel_depth, 1)
		-- b: bias scalar
		-- image_matrix: flattened image matrix w/ shape (image_size * image_size * channel_depth, m)
		-- true_labels: correct "label" array for each image w/ shape (1, m)
		-- interation_count: the number of iterations that the function will loop through during optimization
		-- learning_rate: 
		-- show_cost: print cost value to console every 100 iterations

	Return:
		-- parameters: post-step weight array and bias value
		-- gradients: weight and bias gradients computed through back propagation
		-- costs: cost array holding incremental cost values

	Notes:
		-- Other methods may be used to optimize the weights and bias
	"""

	costs = []

	for i in range(iteration_count):
		gradients, cost = propagate(w, b, image_matrix, true_labels)
		# if math.isnan(cost):
		# 	A = sigmoid(np.dot(w.T, image_matrix) + b)
		# 	print(np.squeeze(A))
		# 	print(cross_entropy_cost(image_matrix.shape[1], A, true_labels))

		dw = gradients['dw']  # obtaining weight gradient from back propagation
		db = gradients['db']  # obtaining bias gradient from back propagation

		w = w - learning_rate * dw  # w array stepping towards local minimum with steps of length: learning_rate
		b = b - learning_rate * db  # b value stepping

		# appends cost value at given iteration increments to costs array for analystics
		collection_rate = 1
		if i % collection_rate == 0:
			costs.append(cost)

		# Shows cost value every 100 iterations if True
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
	Makes a prediction about label using parameters obtained from learning

	Parameters:
		-- w: weights array w/ shape: (image_size * image_size * channel_depth, 3)
		-- b: bias scalar
		-- image_matrix: flattened image matrix w/ shape (image_size * image_size * channel_depth, m)

	Returns:
		-- prediction_labels: numpy array containing prediction labels computed from the activation layer

	Notes:

	"""
	m = image_matrix.shape[1] 					# grab set size again
	prediction_labels = np.zeros((3, m))		# init vector

	activation_layer = sigmoid(np.dot(w.T, image_matrix) + b) # computer sigmoid on prediction data

	# iterates over the activation layer, rounding to the nearest integer, and assigning value to prediction label array
	for i in range(activation_layer.shape[1]):	# covers each data set
		for j in range(3): 						# covers label value within each data set
			if activation_layer[j, i] > 0.5:		# rounding activation value to nearest int (0 or 1)
				prediction_labels[j, i] = 1		# assigning such value to respective location in the prediction label array
			else:								
				prediction_labels[j, i] = 0		# if lower than 0.5, the label is set to False; 0

	# sanity check
	assert(prediction_labels.shape == (3, m))

	return prediction_labels

def model(training_images, training_labels, validation_images, validation_labels, iteration_count, learning_rate, show_cost):
	"""
	Construction of the actual model for training and predicting data

	Parameters:
		-- training_images: 
	Returns:
		-- data:
			costs: the incremental cost value array
			prediction_training_labels: final predictions made by the network on the training data
			prediction_validation_labels: final predication made by the network on the validation data
			original_training_labels: the true labels for the training data
			original_validation_lables: the true labels for the validation data
			w: the final weight array for the network
			b: the final bias value for the network
			learning_rate: the rate at which to step towards a local minimum during gradient descent
			iteration_count: the number of epochs until end


	Notes:
		-- As this is a simple network, only a single bias value and weight array are used. 
		-- More sophisticated networks incorporate several layers of different styles and distinct operators
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

	# Calculates the average proximity of each prediction to the true (normalized)
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














