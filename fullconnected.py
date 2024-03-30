import functions
import numpy as np


class NaNException(Exception):
    "Training process failed, please decrease alpha parameter!"
    pass


class FullConnectedNeuralNetwork():

    def gradient_descent(self, loss, alpha):

        changes_w = []
        changes_b = []
        art = loss.copy()

                
        for i in range(len(self.neurons) + 1):
            if i == 0:
                art = (art @ (self.weights[-1 - i]).T)
            else:
                art = (art @ (self.weights[-1 - i]).T) * self.derivative(self.hidden_outputs_no_activation[-1 - i])

            changes_w.append(alpha * ((self.hidden_outputs_activation[-1 - i]).T @ art) / loss.shape[0])
            changes_b.append(alpha * (np.sum(art) / loss.shape[0]))

            if i == len(self.neurons):
                self.back_loss = (art @ self.weights[0].T)

        for i in range(len(changes_w)):
            self.weights[-2 - i] = self.weights[-2 - i] - changes_w[i]
            self.biases[-1 - i] = self.biases[-1 - i] - changes_b[i]

        try:
            if (np.isnan(self.weights[-2]).sum()) > 0:
                raise NaNException

            return self.back_loss

        except NaNException:
            print("Training process failed, please decrease alpha parameter or number of neurons!")
            raise NaNException


    def __init__(self, n_inputs, neurons, n_outputs, purpose='classification', batches=False):

        self.neurons = neurons
        self.purpose = purpose
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.trained = False

    def prepare(self, gradient_method='gd', activation_func='sigmoid', seed=42, loss_function='cross_entropy_loss',
                optimizer=False):

        np.random.seed(seed)
        self.activation_func, self.derivative = functions.get_func(activation_func)

        self.loss_func, self.loss_derivative = functions.get_loss_func(loss_function)

        if gradient_method == 'gd':
            self.gradient_method = self.gradient_descent
        else:
            raise Exception("gradient method error")

    def forward(self, data):

        if not self.trained:

            for index, neuron in enumerate(self.neurons):

                if index == 0:

                    self.weights = [np.random.uniform(- 0.5, 0.5, size=(self.n_inputs, neuron))]

                    self.biases = [np.random.uniform(- 0.5, 0.5, size=(1, neuron))]

                else:

                    self.weights.append(np.random.uniform(-0.5, 0.5, size=(last, neuron)))

                    self.biases.append(np.random.uniform(-0.5, 0.5, size=(1, neuron)))

                last = neuron + 0

            self.weights.append(np.random.uniform(-0.5, 0.5, size=(last, self.n_outputs)))

            self.biases.append(np.random.uniform(-0.5, 0.5, size=(1, self.n_outputs)))

            self.weights.append(np.eye(self.n_outputs, self.n_outputs))

            self.trained = True

        self.hidden_outputs_no_activation = []
        self.hidden_outputs_activation = []

        self.hidden_outputs_activation.append(data)
        self.hidden_outputs_no_activation.append(data)

        result = data @ self.weights[0] + self.biases[0]

        self.hidden_outputs_no_activation.append(result)
        self.hidden_outputs_activation.append(self.activation_func(result))

        for i in range(len(self.neurons) - 1):
            result = self.hidden_outputs_activation[-1] @ self.weights[i + 1] + self.biases[i + 1]

            self.hidden_outputs_no_activation.append(result)

            self.hidden_outputs_activation.append(self.activation_func(result))

        result = self.hidden_outputs_activation[-1] @ self.weights[-2] + self.biases[-1]

        self.hidden_outputs_no_activation.append(result)

        return result

    def gradient_descent_step(self, loss, alpha):
        return self.gradient_method(loss, alpha)