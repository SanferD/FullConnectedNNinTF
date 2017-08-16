# FullConnectedNNinTF

Fully connected neural networks written in Tensor Flow.

* **cifarnn.py** Neural network with no hidden layer
* **cifarnn1.py** Neural network with 1 hidden layer
* **cifarnn2.py** Neural network with 2 hidden layers

I've written a script **script.py** that will execute all three neural networks on a subset of the CIFAR-100 dataset and will generate plots in the *plots* folder.

One thing you'll have to do is download the CIFAR-100 dataset from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz and unzip it so that it's in the same directory as **script.py**. Doing so will create a folder called **CIFAR-100* and should have three files after unzipping it, namely, *meta*, *test*, and *train*.

cifarnn1.py experiments with a hidden layers of 1/2, 1/4, 1/8, 1/16 hidden nodes as that of the input layer.
cifarnn2.py is similar except that there are two hidden layers. I follow the general rull that the next layer must have fewer nodes than the previous one.

