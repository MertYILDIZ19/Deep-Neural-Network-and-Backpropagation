#-----------------------------------------------------------------------------------
# Implementing a Neural Network In this exercise we will develop a neural
# network with fully-connected layers to perform classification, and test it
# out on the CIFAR-10 dataset.  A bit of setup
#-----------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from two_layernet import TwoLayerNet
from gradient_check import eval_numerical_gradient
from data_utils import get_CIFAR10_data
from vis_utils import visualize_grid
#-------------------------- * End of setup *---------------------------------------

#-------------------------------------------------------
# Some helper functions
# ------------------------------------------------------
def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

#-------------------------- * End of helper functions *--------------------------------



#======================================================================================
# Q1: Implementing forward pass and the loss functions
#======================================================================================
# We will use the class `TwoLayerNet` in the file `two_layernet.py` to
# represent instances of our network.  The network parameters are stored in the
# instance variable `self.params` where keys are string parameter names and
# values are numpy arrays.  Below, we initialize toy data and a toy model that
# we will use to develop your implementation.
#--------------------------------------------------------------------------------------

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()


# Forward pass: compute scores. Open the file `two_layernet.py` and look at the
# method `TwoLayerNet.loss`. This function takes the
# data and weights and computes the per-class scores, the loss, and the gradients
# on the parameters.
#
# Implement the first part of the forward pass which uses the weights and
# biases to compute the scores for all inputs.

scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
 [0.36446210, 0.22911264, 0.40642526],
 [0.47590629, 0.17217039, 0.35192332],
 [0.43035767, 0.26164229, 0.30800004],
 [0.41583127, 0.29832280, 0.28584593],
 [0.36328815, 0.32279939, 0.31391246]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))


# Forward pass: compute loss. In the same function, implement the second part
# that computes the data and regularization loss.
loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))



#======================================================================================
# Q2:Computing gradients using back propogation
#======================================================================================
# Implement the rest of the function. This will compute the gradient of the
# loss with respect to the variables `W1`, `b1`, `W2`, and `b2`. Now that you
# have a correctly implemented forward pass, you can debug your backward pass
# using a numeric gradient check:

# Use numeric gradient checking to check your implementation of the backward
# pass.  If your implementation is correct, the difference between the numeric
# and analytic gradients should be less than 1e-8 for each of W1, W2, b1, and
# b2.
loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))



#======================================================================================
# Q3: Train the network using gradient descent
#======================================================================================

# To train the network we will use stochastic gradient
# descent (SGD). Look at the function `TwoLayerNet.train` and fill in the
# missing sections to implement the training procedure.  You will also have to
# implement `TwoLayerNet.predict`, as the training process periodically
# performs prediction to keep track of accuracy over time while the network
# trains.
# Once you have implemented the method, run the code below to train a
# two-layer network on toy data. You should achieve a training loss less than
# 0.02.

net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=True)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.figure(1)
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# Load the data
# Now that you have implemented a two-layer network that passes
# gradient checks and works on toy data, it's time to load up our favorite
# CIFAR-10 data so we can use it to train a classifier on a real dataset.
# Invoke the get_CIFAR10_data function to get our data.

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some images to get a feel for the data
plt.figure(2)
plt.imshow(visualize_grid(X_train[:100, :].reshape(100, 32,32, 3), padding=3).astype('uint8'))
plt.gca().axis('off')
plt.show()


# Train a network
# To train our network we will use SGD. In addition, we will
# adjust the learning rate with an exponential learning rate schedule as
# optimization proceeds; after each epoch, we will reduce the learning rate by
# multiplying it by a decay rate.

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)
# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)


# Debug the training
# With the default parameters we provided above, you should get a validation
# accuracy of about 0.29 on the validation set, which is not very good.
#
# One strategy for getting insight into what is wrong is to plot the loss
# function and the accuracies on the training and validation sets during
# optimization.
#
# Another strategy is to visualize the weights that were learned in the first
# layer of the network. In most neural networks trained on visual data, the
# first layer weights typically show some visible structure when visualized.


# Plot the loss function and train / validation accuracies
plt.figure(figsize=(20,10))
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()


# Visualize the weights of the network
plt.figure(5)
show_net_weights(net)


# Tune your hyperparameters
#
# **What's wrong?**. Looking at the visualizations above, we see that the loss
# is decreasing more or less linearly, which seems to suggest that the learning
# rate may be too low. Moreover, there is no gap between the training and
# validation accuracy, suggesting that the model we used has low capacity, and
# that we should increase its size. On the other hand, with a very large model
# we would expect to see more overfitting, which would manifest itself as a
# very large gap between the training and validation accuracy.
#
# **Tuning**. Tuning the hyperparameters and developing intuition for how they
# affect the final performance is a large part of using Neural Networks, so we
# want you to get a lot of practice. Below, you should experiment with
# different values of the various hyperparameters, including hidden layer size,
# learning rate, numer of training epochs, and regularization strength. You
# might also consider tuning the learning rate decay, but you should be able to
# get good performance using the default value.
#
# **Approximate results**. You should be aim to achieve a classification
# accuracy of greater than 48% on the validation set. Our best network gets
# over 52% on the validation set.
#
# **Experiment**: You goal in this exercise is to get as good of a result on
# CIFAR-10 as you can (52% could serve as a reference), with a fully-connected
# Neural Network. Feel free to implement your own techniques (e.g. PCA to reduce
# dimensionality, or adding dropout, or adding features to the solver, etc.).

# **Explain your hyperparameter tuning process in the report.**


best_net = None # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
#################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
## PCA ##
from sklearn.decomposition import PCA
pca = PCA(510)
pca.fit(X_train)
ncomponent = pca.n_components_
#
train_img_pca = pca.transform(X_train)
validation_img_pca = pca.transform(X_val)
test_img_pca = pca.transform(X_test)
#
input_size = ncomponent
num_classes = 10
## END OF PCA ##
## START OF HYPERPARAMETER TUNING ##
best_net = None # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
#################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

best_val = -1
best_stats = None
#learning_rates = [1e-1,1e-2,1e-4]
#regularization_strengths = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# learning_rates = [1e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.8e-3, 1.9e-3]
# regularization_strengths = [1e-4, 1e-3, 1e-2]
# hidden_size = [300, 400, 500]
# num_iterations = [1600, 2000, 2500]
# batch_size = [100, 200, 400, 500]
# #hidden_size = 400
# params = [(x, y, hidden, iteration, b) for x in learning_rates for y in regularization_strengths for hidden in hidden_size for iteration in num_iterations for b in batch_size]

# results = {}
# iters = 150
learning_rates = [1e-3, 2.4e-3]
regularization_strengths = [1e-4, 1e-3]
hidden_size = 600
params = [(x,y) for x in learning_rates for y in regularization_strengths ]
results = {}
iters = 10000
for lr, regularization in params:
      net = TwoLayerNet(input_size,hidden_size,num_classes)
      stats = net.train(train_img_pca,y_train,validation_img_pca,y_val,
                        num_iters = 6000,batch_size = 200,learning_rate = lr,
                        learning_rate_decay = 0.90,reg = regularization)
      y_train_pred = net.predict(train_img_pca)
      acc_train = np.mean(y_train == y_train_pred)
      y_val_pred = net.predict(validation_img_pca)
      acc_val = np.mean(y_val == y_val_pred)
      results[(lr,regularization)] = (acc_train,acc_val)
      print("ACCURACY-->>", acc_val)
      if best_val < acc_val:
          best_stats = stats
          best_val = acc_val
          best_net = net
for (lr,reg) in sorted(results):
    (train_accuracy,val_accuracy) = results[(lr,reg)]
    print('lr:%f,reg:%f,train_accuracy:%f,val_accuracy:%f' %(lr,reg,train_accuracy,val_accuracy)) # this line should be fixed.
print('best validation accuracy achieved during cross-validation:%f' %best_val)

## END OF HYPERPARAMETER TUNING ##
# best_val = -1
# best_stats = None
# #learning_rates = [1e-1,1e-2,1e-3,1e-4]
# #regularization_strengths = [0.3,0.4,0.5,0.6]
# learning_rates = [1e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.8e-3]
# regularization_strengths = [1e-4, 1e-3, 1e-2]
# hidden_size = [300, 400]
# num_iterations = [1600]
# batch_size = [400, 500]
# #hidden_size = 400
# params = [(x, y, hidden, iteration, b) for x in learning_rates for y in regularization_strengths for hidden in hidden_size for iteration in num_iterations for b in batch_size]

# results = {}
# iters = 10000
# for lr, regularization, hidden, iteration, b in params:
#       net = TwoLayerNet(input_size,hidden,num_classes)
#       stats = net.train(X_train,y_train,X_val,y_val,
#                         num_iters = iteration,batch_size = b,learning_rate = lr,
#                         learning_rate_decay = 0.90,reg = regularization)
#       y_train_pred = net.predict(X_train)
#       acc_train = np.mean(y_train == y_train_pred)
#       y_val_pred = net.predict(X_val)
#       acc_val = np.mean(y_val == y_val_pred)
#       results[(lr,regularization)] = (acc_train,acc_val)
#       print("ACCURACY-->>", acc_val)
#       if best_val < acc_val:
#           best_stats = stats
#           best_val = acc_val
#           best_net = net
# for (lr,reg) in sorted(results):
#     (train_accuracy,val_accuracy) = results[(lr,reg)]
#     print('lr:%f,reg:%f,train_accuracy:%f,val_accuracy:%f' %(lr,reg,train_accuracy,val_accuracy)) # this line should be fixed.
# print('best validation accuracy achieved during cross-validation:%f' %best_val)

# Plot the loss function and train / validation accuracies
plt.figure(figsize=(20,10))
plt.subplot(2, 1, 1)
plt.plot(best_stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(best_stats['train_acc_history'], label='train',color='r')
plt.plot(best_stats['val_acc_history'], label='val',color='g')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

# END OF NEW PLOTS SECTION


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# visualize the weights of the best network
# plt.figure(6)
# show_net_weights(best_net)


# Run on the test set
# When you are done experimenting, you should evaluate your final trained
# network on the test set; you should get above 48%.

# test_acc = (best_net.predict(X_test) == y_test).mean()
# print('Test accuracy: ', test_acc)

test_acc = (best_net.predict(test_img_pca) == y_test).mean()
print('Test accuracy: ', test_acc)