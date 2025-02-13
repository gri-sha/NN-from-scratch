import os
import numpy as np
import tensorflow as tf

def download_data():
    print("::: Dataset download...")

    if not (os.path.exists("./dataset")):
        os.mkdir("./dataset")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    print("::: Dataset downloaded successfully")
    print()

    np.save("dataset/mnist_x_train.npy", x_train)
    np.save("dataset/mnist_y_train.npy", y_train)
    np.save("dataset/mnist_x_test.npy", x_test)
    np.save("dataset/mnist_y_test.npy", y_test)

    print("::: Dataset saved successfully")
    print()

if __name__ == "__main__":
    if not (
        os.path.exists("dataset/mnist_x_train.npy")
        and os.path.exists("dataset/mnist_y_train.npy")
        and os.path.exists("dataset/mnist_x_test.npy")
        and os.path.exists("dataset/mnist_y_test.npy")
    ):
        download_data()