import tensorflow as tf
# keras is a high level api
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# Our data images are 28x28 pixel values
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Labels are between 0 and 9
# 0:T-shirt/top, 1:Trouser 2:Pullover 3:Dress 4:Coat 5:Sandal 6:Shirt 7:Sneaker 8:Bag 9:Ankle boot

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizing the data
train_images = train_images/255.0
test_images = test_images/255.0

# To show python generated pixels
# plt.imshow(train_images[7])
# To show original image
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

# We must flatten the data, so our 28 x 28 array becomes a 784 x 1 array
# We would like to make 10 output neurons, symbolizing each of the 10 type of clothing
# We would like to have a hidden layer of 128 neurons, that are between the input and output layers to increase
# Complexity of weights and biases, allowing more combinations of patterns to perhaps increase accuracy (arbitrary)

model = keras.Sequential([
    # Flatten data so it is passable to individual neurons
    keras.layers.Flatten(input_shape=(28,28)),
    # Dense layer: Fully connected layer
    # Connect each neuron to the 128 hidden layer neurons
    # Relu (Rectified Linear Unit) is a common activation function that works with a wide variety of
    #   different applications and purposes
    keras.layers.Dense(128, activation="relu"),
    # Softmax is an activation function that picks values for each neuron so that all the values add up to 1
    # (Probability assignment to each output neuron)
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Epochs are "how many times it will see the same image", maybe in a different order for tweaking accuracy
model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print("Tested Acc:", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

# Find the highest probability neuron and send its index corresponding to the classification list
# print(class_names[np.argmax(prediction[0])])
