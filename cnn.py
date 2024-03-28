import numpy as np
import functions
import flatten
import fullconnected 
import convolutional
import pooling
import metrics


class ConvolutionalNeuralNetwork():

    def __init__(self, layers):
        self.prep = []
        # layers = [('conv', [1, 5, 5, 3], 'relu'),
        #  ('conv', [3, 5, 5, 5], 'relu'),
        #  ('conv', [5, 5, 5, 6], 'relu'),
        #  ('flatten', []),
        #  ('full_conn', [1536, [20, 20], 10, 'classification', True, 'gd', 'leaky_relu'])
        #  ])
        for i in range(len(layers)):
            if layers[i][0] == 'conv':
                self.prep.append(
                    convolutional.Conv_Layer((layers[i][1][0], layers[i][1][1], layers[i][1][2]), layers[i][1][3], layers[i][2]))

            elif layers[i][0] == 'pool':
                self.prep.append(
                    pooling.Pooling((layers[i][1][0], layers[i][1][1]), layers[i][1][2]))

            elif layers[i][0] == 'full_conn':
                self.prep.append(
                    fullconnected.FullConnectedNeuralNetwork(int(layers[i][1][0]), layers[i][1][1], int(layers[i][1][2]),
                                                            layers[i][1][3], layers[i][1][4]))
                self.prep[-1].prepare(layers[i][1][5], layers[i][1][6])

            elif layers[i][0] == 'flatten':
                self.prep.append(flatten.Flatten())

    def train(self, train_batches, test_batches, alpha, n_epochs):

        for epoch in range(n_epochs):

            for index, batch in enumerate(train_batches):

                result, temp = batch

                self.alpha = alpha

                result.shape = (result.shape[0], 1, 28, 28)

                for i in range(len(self.prep)):
                    result = self.prep[i].forward(result)

                answer = functions.softmax(result)

                loss = answer - np.int_(np.arange(0, 10) == temp)

                for i in range(len(self.prep)):
                    loss = self.prep[-1 - i].gradient_descent_step(loss, self.alpha)

                if index % 20 == 0:
                    val_acc = []
                    for batch in test_batches:
                        result, temp = batch
                        result.shape = (result.shape[0], 1, 28, 28)

                        for i in range(len(self.prep)):
                            result = self.prep[i].forward(result)

                        val_acc.append(metrics.accuracy(result, np.int_(np.arange(0, 10) == temp)))

                    print('For epoch number: {}, validation accuracy is: {}'.format(epoch, round(
                            np.mean(val_acc), 4)))

    def forward(self, data_image):
        result = data_image

        for i in range(len(self.prep)):
            result = self.prep[i].forward(result)

        answer = functions.softmax(result)
        return answer