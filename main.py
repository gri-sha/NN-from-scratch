# TensorFlow is only used for simplified download of MNIST dataset
import tensorflow as tf
import numpy as np
# Matplotlib is used to visualize the images and labels
import matplotlib.pyplot as plt
import os
from structures import *


def download_data(rand_visual=0):
    print("::: Dataset download...")

    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Check the shape of the data
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    # Normalize the images to a range of 0 to 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    print("::: Dataset downloaded successfully")
    print()

    # Save the datasets as NumPy files
    np.save("dataset/mnist_x_train.npy", x_train)
    np.save("dataset/mnist_y_train.npy", y_train)
    np.save("dataset/mnist_x_test.npy", x_test)
    np.save("dataset/mnist_y_test.npy", y_test)

def visualize(train_data, train_labels, test_data, test_labels, num_train=1, num_test=1):

    total_images = num_train + num_test
    plt.figure(figsize=(15, 5))

    for i in range(num_train):
        rand_index = np.random.randint(0, train_data.shape[0])
        plt.subplot(1, total_images, i + 1)
        plt.imshow(train_data[rand_index], cmap='gray')
        plt.title(f"Train Label: {train_labels[rand_index]}\nIndex: {rand_index}")
        plt.axis('off')

    for i in range(num_test):
        rand_index = np.random.randint(0, test_data.shape[0])
        plt.subplot(1, total_images, num_train + i + 1)
        plt.imshow(test_data[rand_index], cmap='gray')
        plt.title(f"Test Label: {test_labels[rand_index]}\nIndex: {rand_index}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def train(train_data, train_labels, act_type="ReLU", hidden_layers_shape=(32, 32), init_weights_mult=0.01, learning_rate=0.01, mini_batch_size=10, epoch=10):

    if np.shape(train_data)[0] != np.shape(train_labels)[0]:
        raise ValueError("The number of training samples and labels must match.")

    input_size = train_data.shape[1] * train_data.shape[2]

    nn = NeuralNetwork(
        shape=(input_size, *hidden_layers_shape, 10),
        act_type=act_type,
        init_weights_mult=init_weights_mult,
        learning_rate=learning_rate,
    )
    
    for _ in range(epoch):
        accumulated_error = np.zeros((1, 10))

        for i in range(len(train_data)):
            input = train_data[i].reshape((1, input_size))
            output = nn.forward(input)

            true_val = np.zeros((1, 10))
            true_val[0][train_labels[i]] = 1

            error = (output - true_val)
            accumulated_error += error

            if (i + 1) % mini_batch_size == 0:
                nn.backward(accumulated_error)
                accumulated_error = np.zeros((1, 10))

    return nn

def test(test_data, test_labels, nn):
    # TODO: проблема в этой функции
    right_ans = 0
    num_samples = len(test_data)

    for i in range(test_data.shape[0]):
        input_data = test_data[i].reshape((1, test_data.shape[1] * test_data.shape[2]))
        output = nn.forward(input_data)

        predicted_label = np.argmax(output)
        if predicted_label == test_labels[i]:
            right_ans += 1

    accuracy = right_ans / num_samples if num_samples > 0 else 0.0
    print(f'Correct predictions: {right_ans}/{num_samples}')
    print(f'Accuracy: {accuracy:.2%}')

    return accuracy


if __name__ == "__main__":
    # Load the data
    if not (
        os.path.exists("dataset/mnist_x_train.npy")
        and os.path.exists("dataset/mnist_y_train.npy")
        and os.path.exists("dataset/mnist_x_test.npy")
        and os.path.exists("dataset/mnist_y_test.npy")
    ):
        download_data()

    train_data = np.load("dataset/mnist_x_train.npy")
    train_labels = np.load("dataset/mnist_y_train.npy")
    test_data = np.load("dataset/mnist_x_test.npy")
    test_labels = np.load("dataset/mnist_y_test.npy")

    # visualize(train_data, train_labels, test_data, test_labels, num_train=6, num_test=5)
    nn = train(train_data, train_labels, hidden_layers_shape=(16, 16), init_weights_mult=0.01, learning_rate=0.01, act_type="ReLU", mini_batch_size=10, epoch=16)
    print(nn)
    test(test_data, test_labels, nn)
