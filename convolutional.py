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

        # initialize default filter weights
        for i in range(1, self.n_filters):
            self.filter_array = np.append(self.filter_array, [
                np.random.uniform(-1, 1, (self.filter_size[0], self.filter_size[1], self.filter_size[2]))], axis=0)

        self.activation_func, self.derivative = functions.get_func(activation_func)

    def forward(self, image_array):

        self.image = image_array.copy()

        new_image_array = np.zeros((
                                   image_array.shape[0], self.n_filters, image_array.shape[2] - self.filter_size[1] + 1,
                                   image_array.shape[3] - self.filter_size[2] + 1))

        for i in range(image_array.shape[0]):
            for j in range(self.n_filters):
                # we use here scipy.signal.fftconvolve, not np.convolve, because it is usually faster. look here: https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
                new_image_array[i][j] = np.squeeze(
                    scipy.signal.fftconvolve(image_array[i], self.filter_array[j], mode='valid'), axis=0) + self.b[j]

        return self.activation_func(new_image_array)

    def gradient_descent_step(self, loss, alpha):

        loss = loss @ self.derivative(loss) / loss.shape[0]

        d_image = np.zeros(self.filter_array.shape)

        d_ = np.zeros(self.b.shape)

        temp2 = np.zeros(self.image.shape[0])
        temp = np.zeros((self.image.shape[0], *self.filter_size[1:]))

        for i in range(self.n_filters):
            for j in range(self.filter_size[0]):
                for k in range(self.image.shape[0]):
                    temp[k] = scipy.signal.fftconvolve(self.image[i][j], loss[k][i], mode='valid')
                d_image[i][j] = temp.mean(axis=0)

        for i in range(self.n_filters):
            for j in range(self.image.shape[0]):
                temp2[j] = np.sum(loss[i])
            d_[i] = temp2.mean()


        rot_filter_array = np.zeros(self.filter_array.shape)

        for i in range(self.filter_array.shape[0]):
            rot_filter_array[i] = np.rot90(np.rot90(self.filter_array[i], -1, (1, 2)), -1, (1, 2))

        padded = np.pad(loss, ((0, 0), (0, 0), (self.filter_size[1] - 1, self.filter_size[1] - 1),
                               (self.filter_size[1] - 1, self.filter_size[1] - 1)), 'constant', constant_values=(0))

        new_loss = np.zeros(self.image.shape)

        temp = np.zeros((self.n_filters, self.image.shape[2] , self.image.shape[3]))

        for i in range(padded.shape[0]):
            for k in range(self.image.shape[1]):
                for j in range(rot_filter_array.shape[0]):
                    temp[j] = scipy.signal.fftconvolve(padded[i][j], rot_filter_array[j][k], mode='valid')
            new_loss[i][k] = np.mean(temp, axis=0)

        self.filter_array = self.filter_array - alpha * d_image
        self.b = self.b - alpha * d_

        try:
            if (np.isnan(self.filter_array).sum()) > 0:
                raise NaNException
            return new_loss

        except NaNException:
            print("Training process failed, please decrease alpha parameter!")
            raise NaNException