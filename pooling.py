import numpy as np


class Pooling():

    def __init__(self, size, p_type='max'):

        self.size = size
        if p_type == 'min':
            self.p_type = np.min
        elif p_type == 'max':
            self.p_type = np.max
        elif p_type == 'average':
            self.p_type = np.mean
        else:
            raise 'Type error'

    def forward(self, image_array):

        array = image_array.copy()

        result_full = np.zeros(
            (array.shape[0], array.shape[1], int(array.shape[2] / self.size[0]), int(array.shape[3] / self.size[1])))

        for k in range(array.shape[0]):
            for m in range(array.shape[1]):
                # temp = np.squeeze(array[i], axis = 0)
                temp = array[k][m]
                result = []
                self.i = 0
                while self.i < temp.shape[0] - self.size[0] + 1:
                    self.j = 0
                    while self.j < temp.shape[1] - self.size[1] + 1:
                        result.append(self.p_type(temp[self.i:self.i + self.size[0], self.j:self.j + self.size[1]]))
                        temp[self.i:self.i + self.size[0], self.j:self.j + self.size[1]] = (temp[
                                                                                            self.i:self.i + self.size[
                                                                                                0],
                                                                                            self.j: self.j + self.size[
                                                                                                1]]) * [temp[
                                                                                                        self.i:self.i +
                                                                                                               self.size[
                                                                                                                   0],
                                                                                                        self.j:self.j +
                                                                                                               self.size[
                                                                                                                   1]] >= self.p_type(
                            temp[self.i:self.i + self.size[0], self.j:self.j + self.size[1]])]

                        self.j += self.size[1]
                    self.i += self.size[0]

                result_full[k][m] = np.expand_dims(
                    np.array(result).reshape(int(temp.shape[0] / self.size[0]), int(temp.shape[1] / self.size[1])),
                    axis=0)
                array[k][m] = np.expand_dims(temp, axis=0)

        self.array = array
        return result_full

    def gradient_descent_step(self, back_prop_array, alpha):
        un_pooled_array = self.array.copy()

        for k in range(self.array.shape[0]):

            for m in range(self.array.shape[1]):

                temp = self.array[k][m]

                i = 0

                while self.i < temp.shape[0] - self.size[0] + 1:
                    j = 0
                    self.j = 0
                    while self.j < temp.shape[1] - self.size[1] + 1:
                        temp[self.i:self.i + self.size[0], self.j:self.j + self.size[1]] = temp[
                                                                                           self.i:self.i + self.size[0],
                                                                                           self.j: self.j + self.size[
                                                                                               1]] * \
                                                                                           back_prop_array[k][m][i][j]

                        self.j += self.size[1]
                        j += 1
                    self.i += self.size[0]
                    i += 1
                un_pooled_array[k][m] = np.expand_dims(temp, axis=0)

        return un_pooled_array