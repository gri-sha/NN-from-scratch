import numpy as np


def linear(z, a=1, b=0):
    return a*z + b


def linear_derivative(acts, a=1):
    return np.ones(shape=acts.shape) * a


def relu(z):
    return np.maximum(0, z)


def relu_derivative(acts):
    return np.where(acts > 0, 1.0, 0.0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(acts):
    return sigmoid(acts) * (1 - sigmoid(acts))


def tanh(z):
    return np.tanh(z)

def tanh_derivative(acts):
    return 1 - np.tanh(acts) ** 2


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_derivative(acts):
    # Jacobian matrix does not let the calculations to be simple
    jacobian_m = np.diag(acts)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = acts[i] * (1 - acts[i])
            else: 
                jacobian_m[i][j] = -acts[i] * acts[j]
    return jacobian_m


VALID_ACT = {
    "linear": (linear, linear_derivative),
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    # "softmax": (softmax, softmax_derivative),
}
