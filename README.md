# Deep-Neural-Network-and-Backpropagation

Deep neural networks have shown staggering performances in various learning tasks, including computer vision,
natural language processing, and sound processing. They have made the model designing more flexible by
enabling end-to-end training.
In this exercise, we get to have a first hands-on experience with neural network training. Many frameworks (e.g.
PyTorch, Tensorflow, Caffe) allow easy usage of deep neural networks without precise knowledge on the inner
workings of backpropagation and gradient descent algorithms. While these are very useful tools, it is important
to get a good understanding of how to implement basic network training from scratch, before using this libraries
to speed up the process. For this purpose we will implement a simple two-layer neural network and its training
algorithm based on back-propagation using only basic matrix operations in questions 1 to 3. In question 4, we
will use a popular deep learning library, PyTorch, to do the same and understand the advantages offered in
using such tools.
As a benchmark to test our models, we consider an image classification task using the widely used CIFAR-10
dataset. This dataset consists of 50000 training images of 32x32 resolution with 10 object classes, namely
airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The task is to code and train a
parameterised model for classifying those images. This involves
• Implementing the feedforward model (Question 1).
• Implementing the backpropagation algorithm (gradient computation) (Question 2).
• Training the model using stochastic gradient descent and improving the model training with better hyperparameters (Question 3).
• Using the PyTorch Library to implement the above and experiment with deeper networks (Question 4).
A note on notation: throughout the exercise, notation vi
is used to denote the i
th element of vector v
Questions 1-3 are based on the script ex2 FCnet.py and question 4 is based on the script ex2 pytorch.py.
To download the CIFAR-10 dataset please execute the script datasets/get datasets.sh or use the function torchvision.datasets.CIFAR10, as illustrated in the section “Load the CIFAR-10 dataset” in the file
ex2 pytorch.py.
