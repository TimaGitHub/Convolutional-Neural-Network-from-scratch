import numpy as np


class Flatten():
    def __init__(self):
        pass

    def gradient_descent_step(self, loss, alpha):
        return loss.reshape(loss.shape[0], self.width, self.size, self.size)

    def forward(self, data_image):
        self.width = data_image.shape[1]
        self.size = data_image.shape[2]

        return data_image.reshape(data_image.shape[0], -1)