A repository containing a clothing classification model using neural networks.

Introduction:

Retrieves pixel images of clothings, size 28x28 (reconfigurable), and outputs the type of clothing it most likely classifies in (ie: T-shirt, Trouser, Sandal, Coat, Sneaker...)

Design description:

1) Normalizes pixel values, as each pixel is described as a double value in the range 0 to 255. We wish to convert it into a range between 0 and 1.

2) Flattens input data, an array in the form 28x28, into an array of the form 784x1 -> 784 input neurons

3) Selects arbitrary number of neurons for the hidden layer that reaches a high level of accuracy (128 hidden layer neurons chosen)

4) Selects activation functions for all neurons in the hidden layer (ReLU activation function chosen)

5) Initializes 10 output neurons, one for each type of clothing classification, and set their activation functions to be softmax

6) Fits model on training data with selected number of epochs (5)

Libraries utilized: Tensorflow, keras, numpy, matplotlib
Dataset utilized: keras.datasets.fashion_mnist
loss: 0.2972
Accuracy achieved: 89.07% (Higher accuracy may be achieved through changing hyperparameters, such as # of neurons, # of epochs, and activation function selections)
