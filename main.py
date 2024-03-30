from dataloader import DataBatcher
from cnn import ConvolutionalNeuralNetwork
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('train.csv')

    data = np.array(data)
    data = data.astype(float)
    np.random.shuffle(data)
    test_data = data[0:5000]
    train_data = data[5000: 42000]
    test_data[:, 1:] = test_data[:, 1:] / 255
    train_data[:, 1:] = train_data[:, 1:] / 255

    test_batches = DataBatcher(test_data, 64, True)
    train_batches = DataBatcher(train_data, 64, True)

    test = ConvolutionalNeuralNetwork([('conv', [1, 5, 5, 3], 'relu'),     # 64x1x28x28 -> 64x3x24x24
                                       ('conv', [3, 5, 5, 3], 'relu'),     # 64x3x24x24 -> 64x3x20x20
                                       ('pool', [2, 2, 'max']),            # 64x3x20x20 -> 64x3x10x10
                                       ('flatten', []),                    # 64x3x10x10 -> 64x300
                                       ('full_conn', [300, [30, 20], 10,
                                                      'classification',
                                                      True, 'gd',
                                                      'leaky_relu'])       # 64x300 -> 64x10
                                       ])
    test.cosmetic(progress_bar=False, loss_display=True, loss_graphic = False, iterations= 20)

    test.train(train_batches, test_batches, 0.05, 1)
