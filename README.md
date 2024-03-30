# A Convolutional Neural Network From Scratch
This project was made for educational purposes to practice the skills and knowledge gained during the deep learning course and build my own convolutional neural network using minimum number of packages (numpy, pandas, matplotlib, scipy).

This neural network is for classifications tasks and it was mostly built for digit dataset from [kaggle](https://www.kaggle.com/competitions/digit-recognizer/overview) (**train.csv**, **test.csv** files).

# Usage
Open main.py file in any notebook or ide. 

``` python
test = ConvolutionalNeuralNetwork([('conv', [1, 5, 5, 3], 'relu'),         # 64x1x28x28 -> 64x3x24x24
                                   ('conv', [3, 5, 5, 3], 'relu'),         # 64x3x24x24 -> 64x3x20x20
                                   ('pool', [2, 2, 'max']),                # 64x3x20x20 -> 64x3x10x10
                                   ('flatten', []),                        # 64x3x10x10 -> 64x300
                                   ('full_conn', [300, [30, 20], 10,
                                                      'classification',
                                                       True, 'gd',
                                                      'leaky_relu'])       # 64x300 -> 64x10
                                       ])

test.cosmetic(progress_bar=False, loss_display=True, loss_graphic = False, iterations= 20)

test.train(train_batches, test_batches, 0.05, 3)

```

As you can see, you may choose layer type, convolution filter size, the number of filters, the number of inputs, outputs, hidden layers and number of neurons for every layer, gradient descent algorithm, activation function, alpha parameter etc.

**Note: This implementation of a neural network is scalable, unlike other user implementations.**

However, be careful when you increase the number of layers and neurons, as due to the high losses, the learning process becomes less controllable.

On digit image dataset CNN perfoms well ( *about 90 accuracy* ), but such architecture is not adapted for real tasks, because convolution operation and pooling are **more complex** operations than matrix multiplication. So, the training process takes a lot of time.

## Neural Net Architecture
```python
 ('conv', [1, 5, 5, 3]), ('conv', [3, 5, 5, 3]), ('pool', [2, 2]),  ('flatten'), ('full_conn',[300, [30, 20], 10])
 ```
 ![gh4](https://github.com/TimaGitHub/Neural-Network-from-Scratch/assets/70072941/b454d716-ef84-428f-b412-c8d36fafa717)

 ### progress_bar and loss_display
 ![gh1](https://github.com/TimaGitHub/NeuralNetwork-from-Scratch/assets/70072941/d4484b22-655b-437a-a53f-897ebad3b8f2)

 ### loss_graphic
 ![gh3](https://github.com/TimaGitHub/NeuralNetwork-from-Scratch/assets/70072941/14317df1-68cf-4086-b107-e79e9dbbf55e)




## To-Do List
- [ ] add regularization
- [ ] make it more robust for large number of layers and neurons
- [ ] make it faster
- [ ] make class more pytorch-like
- [ ] add the ability to save and load model parameters


## References

- very clear [explanation](https://colab.research.google.com/drive/1ZMu6C3ZEt3kCSDBNWGM6sicXL5-EhSve?usp=sharing) of how convolutional neural network is made
- nice 3Blue1Brown [playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=vZ3tJjTqXa9iSfBE) about neural networks 
  
- 3Blue1Brown [video](https://youtu.be/KuXjwB4LzSA?si=KJHdPrJK_1tBuZl_) about convolution 

- cool [playlist](https://youtube.com/playlist?list=PL1sQgSTcAaT7MbcLWacjsqoOQvqzMdUWg&si=gCke_NmYGIwUbJ9X) about to understand the whole process

- very important [playlist](https://youtube.com/playlist?list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu&si=XDupIIUFmAu2olXnabout) about propagation in CNN

- one more [explanation](https://youtu.be/m8pOnJxOcqY?si=VuHoljUq4rbAelv6)

- interesting [github project](https://github.com/vzhou842/cnn-from-scratch) on CNN too
