# fruit-classifier
Fruit Classifier, a single-layered Neural Network used to classify various forms of fruit from image input.

## About:
* The project was divided into three sections: data loading, training and predicting, and analysis.
* Data loading is handled by `utils.py`, which reads and formats local training data for the model.
  * Images of any dimension are squished or stretched to fit the desired training dimensions.
  * Red, green, and blue channels are normalized monotonically, and later flattened to a single vector. 
* `classifier.py` handles computation and model creation, forming a single-layered NN.
  * The network uses the **Sigmoid Activation** function and the **Cross-Entropy Cost** function for forward propogation.
  * Back propogation is handled by the **Gradient Descent** function.
  
## Resources used:
* [NumPy](http://www.numpy.org/): A scientific computing package for Python. 
* [scikit-learn](http://scikit-learn.org/): A machine learning library in Python. 
* [Matplotlib](https://matplotlib.org/): A plotting library for Python.

## Example:
* 300 total fruit images used (apples, bananas, and oranges).
* 1000 iterations; learning rate of 0.5%; 128x128 images used
![](https://github.com/michaelsterpka/fruit-classifier/blob/master/example_output/figure_1.png "Example of an apple prediction")
![](https://github.com/michaelsterpka/fruit-classifier/blob/master/example_output/figure_2.png "Example of an orange prediction")
![](https://github.com/michaelsterpka/fruit-classifier/blob/master/example_output/figure_3.png "Example of an banana prediction")

  
## TODO:
* optimize data_set loader
* complete activation functions
* build layers
* provide basic ui for results
* allow for custom image input for testing

