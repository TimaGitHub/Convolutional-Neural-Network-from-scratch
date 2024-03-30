import numpy as np


class Pooling():
    def __init__(self, size, p_type='max'):

        self.size = size
        if p_type == 'min':
            self.p_type = np.min
        elif p_type == 'max':
            self.p_type = np.max
        else:
            raise 'Type error'

    def forward(self, image_array):

        if image_array.shape[2] % self.size[0] != 0:
            raise Exception("Can't apply pooling due to the size, please change it")
        array = image_array.copy()
        result_full = np.zeros(
            (array.shape[0], array.shape[1], int(array.shape[2] / self.size[0]), int(array.shape[3] / self.size[1])))

        for k in range(array.shape[0]):
            for m in range(array.shape[1]):
                # temp = array[k][m]
                result = []
                self.i = 0
                while self.i < array[k][m].shape[0] - self.size[0] + 1:
                    self.j = 0
                    while self.j < array[k][m].shape[1] - self.size[1] + 1:
                        result.append(
                            self.p_type(array[k][m][self.i:self.i + self.size[0], self.j:self.j + self.size[1]]))
                        array[k][m][self.i:self.i + self.size[0], self.j:self.j + self.size[1]] = (array[k][m][
                                                                                                   self.i:self.i +
                                                                                                          self.size[0],
                                                                                                   self.j: self.j +
                                                                                                           self.size[
                                                                                                               1]]) * [
                                                                                                      array[k][m][
                                                                                                      self.i:self.i +
                                                                                                             self.size[
                                                                                                                 0],
                                                                                                      self.j:self.j +
                                                                                                             self.size[
                                                                                                                 1]] == self.p_type(
                                                                                                          array[k][m][
                                                                                                          self.i:self.i +
                                                                                                                 self.size[
                                                                                                                     0],
                                                                                                          self.j:self.j +
                                                                                                                 self.size[
                                                                                                                     1]])]

                        self.j += self.size[1]
                    self.i += self.size[0]

                # result_full[k][m] = np.expand_dims(np.array(result).reshape(int(array[k][m].shape[0] / self.size[0]), int(array[k][m].shape[1] / self.size[1])) , axis = 0)
                result_full[k][m] = np.array(result).reshape(int(array[k][m].shape[0] / self.size[0]),
                                                             int(array[k][m].shape[1] / self.size[1]))
                # array[k][m] =  np.expand_dims(temp, axis = 0)

        self.array = array
        return result_full
    def gradient_descent_step(self, back_prop_array, alpha):
        new_shape = np.zeros(self.array.shape)
        for k in range(self.array.shape[0]):
            for m in range(self.array.shape[1]):
                inx_ = 0
                inx__ = 0
                self.i = 0
                while self.i < self.array[k][m].shape[0] - self.size[0] + 1:
                    self.j = 0
                    inx__ = 0
                    while self.j < self.array[k][m].shape[1] - self.size[1] + 1:
                        new_shape[k][m][self.i:self.i + self.size[0], self.j:self.j + self.size[1]] = \
                        back_prop_array[k][m][inx_][inx__]
                        inx__ += 1
                        self.j += self.size[1]

                    inx_ += 1
                    self.i += self.size[0]

        return np.squeeze([self.array > 0] * new_shape, axis=0)