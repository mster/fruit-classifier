import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

# training data loader function for preparing and correctly labeling training data
# training_path: the path which contains all class subdirectories
# image_size: the height and width value for which the image will be formatted. 
# 		      Resulting images will be square.
# classes: the array of classes to be loaded and formatted for training. 
def load_training_data(training_path, image_size, classes):
	images = [] #np array representation of training images. shape = (image_count, image_size, image_size, 3)
	labels = [] #labels np array (designating which class the image is)
	image_names = [] #image filename np array
	class_set = [] #class name np array

	#image processing loop
	for _class in classes: #iterate through given class array 
		class_index = classes.index(_class) #setting index var to position of array in which the classname resides.
		class_path = os.path.join(training_path, _class, '*g') # *g: general format
		dir_glob = glob.glob(class_path) #stores all files within class_path directory as an glob
		for file in dir_glob: #iterating through files in class_path directory
			image = cv2.imread(file) #loads original image
			image = cv2.resize(image, (image_size, image_size), cv2.INTER_CUBIC) #formats image size using linear interpolation
			image = image.astype(np.float64) #set image value type to float (used in next step)
			image = np.multiply(image, 1.0 / 255.0) #normalizes the RBG values associated with each pixel in the image
			images.append(image) #adds formatted image to images aray
			class_label = np.zeros(len(classes)) #init zero np.array with length equal to class count
			class_label[class_index] = 1.0 #sets specific class value as true. (rest of np.array is 0: 'false')
			labels.append(class_label) #adding class label
			file_name = os.path.basename(file) #gets specific image file name from tail of path
			image_names.append(file_name) #appends image file name
			class_set.append(_class) #appends class name 

	# formatting for return		
	images = np.array(images)
	labels = np.array(labels)
	image_names = np.array(image_names)
	class_set = np.array(class_set)

	return images, labels, image_names, class_set

# DataSet object for storing the several np.arrays associated with the training data
# images: the array of images to store
# labels: the array of respective labels
# image_names: the array of respective image-file names
# class_set: the array of respective class labels
class DataSet(object):
	def __init__(self, images, labels, image_names, class_set):
		self._images = images
		self._labels = labels
		self._image_names = image_names
		self._class_set = class_set

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def image_names(self):			
		return self._image_names

	@property
	def class_set(self):
		return self._class_set

# Data loadinging and formatting function.
# training_path: the path in which the class sub-directories reside.
# image_size: the new height and width of the training images.
# validation_size: the portion of the image set to be used as validation data
def read_data_sets(training_path, image_size, classes, validation_size):
	#for storing training and validation set in a single object
	class DataSets(object):
		pass
	data_sets = DataSets() #init

	# loads training data and formats images appropriately 
	images, labels, image_names, class_set = load_training_data(training_path, image_size, classes)

	#shuffles arrays to prevent overfitting (labels are kept in respective indices)
	images, labels, image_names, class_set = shuffle(images, labels, image_names, class_set) 

	# check if validation_size is a float
	if isinstance(validation_size, float):
		# validation_size is multiplied by the total amount of images to produce bounds for splicing
		validation_size = int(validation_size * images.shape[0]) 

    # splice portion of dataset into training set
	training_images = images[validation_size:]
	training_labels = labels[validation_size:]
	training_image_names = image_names[validation_size:]
	training_class_set = class_set[validation_size:]

	# splice portion of dataset into validation set
	validation_images = images[:validation_size]
	validation_labels = labels[:validation_size]
	validation_image_names = image_names[:validation_size]
	validation_class_set = class_set[:validation_size]

	# creates DataSet objects for training and validation data
	data_sets.train = DataSet(training_images, training_labels, training_image_names, training_class_set)
	data_sets.valid = DataSet(validation_images, validation_labels, validation_image_names, validation_class_set)

	return data_sets






