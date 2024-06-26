import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return x * (1 + np.sign(x)) / 2

def derivative_relu(x):
    return (1 + np.sign(x)) / 2

def leaky_relu(x):
    return x * ((1 + np.sign(x)) / 2 + 0.2 * (1 + np.sign(-x)) / 2)

def derivative_leaky_relu(x):
    return ((1 + np.sign(x)) / 2 + 0.2 * (1 + np.sign(-x)) / 2)

def tanh(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

def derivative_tanh(x):
    return 1 - tanh(x)

def cross_entropy_loss(predicted, true):

    true = np.int_(np.arange(0, 10) == true)

    return -1 * np.sum(true * np.log(predicted), axis=1)

def cross_entropy_loss_derivative(predicted, true):

    true = np.int_(np.arange(0, 10) == true)

    return predicted - true

def softmax(z):
    if z.ndim == 1:
        return np.exp(z) / np.sum(np.exp(z))
    else:
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

def get_func(func_):
    if func_ == 'sigmoid':
        return sigmoid, derivative_sigmoid

    elif func_ == 'relu':
        return relu, derivative_relu

    elif func_ == 'leaky_relu':
        return leaky_relu, derivative_leaky_relu

    elif func_ == 'tanh':
        return tanh, derivative_tanh

    else:
        raise Exception("Activation function is not specified or unknown")

def get_loss_func(func_):
    if func_ == 'cross_entropy_loss':
        return cross_entropy_loss, cross_entropy_loss_derivative
    else:
        raise Exception("Loss function is not specified or unknown")