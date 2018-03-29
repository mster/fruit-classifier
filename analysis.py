import numpy as np
import matplotlib.pyplot as pyplot
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import json

# importing classifier resources
import classifier

# fetch data from model after training
result = classifier.train()

# static array variants
APPLE = [1,0,0]
ORANGE = [0,1,0]
BANANA = [0,0,1]

# misc printing data
correct_counter = 0
show_prediction = True

for i in range(result["prediction_training_labels"].shape[1]):
	guess_array = result["prediction_training_labels"][:,i].astype(int)
	correct_label = result["original_training_labels"][i]

	prediction_label = 'None'
	if np.array_equal(guess_array, APPLE):
		prediction_label = 'apple'
	elif np.array_equal(guess_array, ORANGE):
		prediction_label = 'orange'
	elif np.array_equal(guess_array, BANANA):
		prediction_label = 'banana'

	if(str(prediction_label) == str(correct_label)):
		correct_counter += 1

	if show_prediction:
		print("Prediction: %s Actual: %s Guess Array: %s" % (prediction_label,correct_label,guess_array))

print("Correctly guessed: %s out of %s" % (correct_counter,result["original_training_labels"].shape[0]))



