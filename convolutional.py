import functions
import numpy as np
import scipy.signal


class NaNException(Exception):
    "Training process failed, please decrease alpha parameter!"
    pass

class Conv_Layer():

    def __init__(self, filter_size, n_filters, activation_func='relu'):  # filter_size(z,x,y)

        self.filter_size = filter_size

        self.n_filters = n_filters

        self.b = np.random.uniform(-1, 1, (self.n_filters))

        self.filter_array = np.array(
            [np.random.uniform(-1, 1, (self.filter_size[0], self.filter_size[1], self.filter_size[2]))])

        for i in range(1, self.n_filters):
            self.filter_array = np.append(self.filter_array, [
                np.random.uniform(-1, 1, (self.filter_size[0], self.filter_size[1], self.filter_size[2]))], axis=0)

        if activation_func == 'sigmoid':
            self.activation_func = functions.sigmoid
            self.derivative = functions.derivative_sigmoid

        elif activation_func == 'relu':
            self.activation_func = functions.relu
            self.derivative = functions.derivative_relu

        elif activation_func == 'leaky_relu':
            self.activation_func = functions.leaky_relu
            self.derivative = functions.derivative_leaky_relu

        elif activation_func == 'tanh':
            self.activation_func = functions.tanh
            self.derivative = functions.derivative_tanh

        else:
            raise Exception("Activation function error")

    def forward(self, image_array):

        self.image = image_array.copy()

        new_image_array = np.zeros((
                                   image_array.shape[0], self.n_filters, image_array.shape[2] - self.filter_size[1] + 1,
                                   image_array.shape[3] - self.filter_size[2] + 1))

        for i in range(image_array.shape[0]):

            for j in range(self.n_filters):
                new_image_array[i][j] = np.squeeze(
                    scipy.signal.fftconvolve(image_array[i], self.filter_array[j], mode='valid'), axis=0) + self.b[j]

        return self.activation_func(new_image_array) / new_image_array.max()

    def gradient_descent_step(self, loss, alpha):

        loss = loss @ self.derivative(loss) / loss.shape[0]

        d_image = np.zeros(self.filter_array.shape)

        d_ = np.zeros(self.b.shape)

        temp = np.zeros((self.image.shape[0], self.filter_array.shape[1], self.filter_array.shape[2],
                         self.filter_array.shape[3]))

        temp2 = np.zeros(self.image.shape[0])

        for k in range(self.n_filters):
            for i in range(self.image.shape[0]):
                temp[i] = scipy.signal.fftconvolve(self.image[i], np.expand_dims(loss[i][k], axis=0), mode='valid')
                temp2[i] = np.sum(loss[k])

            d_image[k] = temp.mean(axis=0)

            d_[k] = temp2.mean()

        rot_filter_array = np.zeros(self.filter_array.shape)

        for i in range(self.filter_array.shape[0]):
            rot_filter_array[i] = np.rot90(np.rot90(self.filter_array[i], -1, (1, 2)), -1, (1, 2))

        padded = np.pad(loss, ((0, 0), (0, 0), (self.filter_size[1] - 1, self.filter_size[1] - 1),
                               (self.filter_size[1] - 1, self.filter_size[1] - 1)), 'constant', constant_values=(0))

        new_loss = np.zeros(self.image.shape)
        temp = np.zeros((self.n_filters, rot_filter_array[0].shape[0], self.image.shape[2], self.image.shape[3]))
        for i in range(padded.shape[0]):
            for k in range(self.n_filters):
                _ = np.pad(np.expand_dims(padded[i][k], axis=0),
                           ((rot_filter_array[k].shape[0] - 1, rot_filter_array[k].shape[0] - 1), (0, 0), (0, 0)),
                           'constant', constant_values=(0))

                temp[k] = scipy.signal.fftconvolve(_, rot_filter_array[k], mode='valid')

            new_loss[i] = np.mean(temp, axis=0)

        self.filter_array = self.filter_array - alpha * d_image
        self.b = self.b - alpha * d_

        try:
            if (np.isnan(self.filter_array).sum()) > 0:
                raise NaNException

            return new_loss

        except NaNException:
            print("Training process failed, please decrease alpha parameter!")
            raise NaNException