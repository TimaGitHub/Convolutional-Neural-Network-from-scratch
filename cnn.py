import numpy as np
import functions
import flatten
import fullconnected 
import convolutional
import pooling
import metrics
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib
matplotlib.use("TkAgg")

class ConvolutionalNeuralNetwork():

    #  example
    #  layers = [
    #  ('conv', [input dimension, filter size, filter size, number of filters], 'relu'),

    #  note:
    #  for the first convolutional layer input_dimension = 1, because image format is grayscale (x and y, z = 1 (default))
    #  for the next layers: input dimension = number of filters

    #  ('conv', [input dimension, filter size, filter size, number of filters], 'relu'),
    #  ('pool', [window size, window size, 'max' or 'min']),
    #  ('flatten', []),
    #  ('full_conn', [input number of images (based on previous layers) , [number of neurons, number of neurons, ......], output , 'classification', True, 'gd', 'leaky_relu'])
    #  ]
    def __init__(self, layers):
        self.prep = []
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

        self.history_scores = []

        for epoch in self.tqdm(range(n_epochs), position=0, leave=True):

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

                if self.loss_display and index % self.iterations == 0:
                    val_acc = []
                    for batch in test_batches:
                        result, temp = batch
                        result.shape = (result.shape[0], 1, 28, 28)

                        for i in range(len(self.prep)):
                            result = self.prep[i].forward(result)

                        val_acc.append(metrics.accuracy(result, np.int_(np.arange(0, 10) == temp)))

                    print('For epoch number: {}, validation accuracy is: {}'.format(epoch, round(
                            np.mean(val_acc), 4)))

                    self.history_scores.append(np.mean(val_acc))

        if self.loss_graphic:

            fig, ax1 = plt.subplots(figsize=(9, 8))

            clear_output(True)

            ax1.set_xlabel('iters')

            ax1.set_ylabel('accuracy', color='blue')

            t = np.arange(len(self.history_scores))

            ax1.plot(t, self.history_scores)

            plt.locator_params(axis='y', nbins=40)

            fig.tight_layout()

            plt.show()

    def forward(self, data_image):
        result = data_image

        for i in range(len(self.prep)):
            result = self.prep[i].forward(result)

        answer = functions.softmax(result)
        return answer

    def cosmetic(self, progress_bar=False, loss_display=False, loss_graphic = False, iterations=0):
        # printing this "For epoch number: ..., validation accuracy is: ..., loss is ..."
        self.loss_display = loss_display

        # to depict the learning process.
        self.loss_graphic = loss_graphic

        # how often you would like to get message about training process
        self.iterations = iterations

        if not progress_bar:
            def tqdm_False(x, **params__):
                return x

            self.tqdm = tqdm_False
        else:
            self.tqdm = tqdm