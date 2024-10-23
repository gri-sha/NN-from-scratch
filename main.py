# TensorFlow is only used for simplified download of MNIST dataset
import tensorflow as tf
import numpy as np
# Matplotlib is used to visualize the images and labels
import matplotlib.pyplot as plt
import os
from structures import *


def download_data(rand_visual=0):
    print("::: Dataset download...")

    if not (os.path.exists("./dataset")):
        os.mkdir("./dataset")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    # Normalization
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    print("::: Dataset downloaded successfully")
    print()

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


if __name__ == "__main__":
    if not (
        os.path.exists("dataset/mnist_x_train.npy")
        and os.path.exists("dataset/mnist_y_train.npy")
        and os.path.exists("dataset/mnist_x_test.npy")
        and os.path.exists("dataset/mnist_y_test.npy")
    ):
        download_data()

    x_train = np.load("dataset/mnist_x_train.npy")
    y_train = np.load("dataset/mnist_y_train.npy")
    x_test = np.load("dataset/mnist_x_test.npy")
    y_test = np.load("dataset/mnist_y_test.npy")

    # visualize(train_data, train_labels, test_data, test_labels, num_train=6, num_test=5)

    # Best option found so far
    model = DenseNetwork(
        layers=[
            DenseLayer(784, act_type="relu"),
            DenseLayer(64, act_type="tanh"),
            DenseLayer(32, act_type="relu"),
            DenseLayer(10, act_type="relu"),
        ],
        init_weights_mult=0.01,
        learning_rate=0.01
    )

    print(model)
    model.fit(train_data=x_train, train_labels=y_train, mini_batch_size=10, epochs=8)
    print(model)
    model.test(test_data=x_test, test_labels=y_test)
