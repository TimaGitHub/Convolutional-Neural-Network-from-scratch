
# Press the green button in the gutter to run the script.

import dataloader
import cnn
import pandas as pd
import numpy as np


if __name__ == '__main__':
    data = pd.read_csv('train.csv')

    data = np.array(data)
    data = data.astype(float)

    test_data = data[0:1000]
    train_data = data[1000: 10000]
    test_data[:, 1:] = test_data[:, 1:] / 255
    train_data[:, 1:] = train_data[:, 1:] / 255

    test_batches = dataloader.DataBatcher(test_data, 64, True)
    train_batches = dataloader.DataBatcher(train_data, 64, True)

    test = cnn.ConvolutionalNeuralNetwork([('conv', [1, 3, 3, 3], 'relu'),
                                       ('conv', [3, 3, 3, 5], 'relu'),
                                       ('pool', [2, 2, 'max']),
                                       ('flatten', []),
                                       ('full_conn', [720, [20, 20], 10, 'classification', True, 'gd', 'leaky_relu'])
                                       ])

    test.train(train_batches, test_batches, 0.01, 3)