The goal of this assignment is to try performing automatic hyperparameter
search. Your goal is to optimize [conv_net.py](https://github.com/ufal/npfl114/tree/master/labs/13/conv_net.py)
model with several hyperparameters, so that it achieves highest validation
accuracy on Fashion MNIST dataset after two epochs of training.
The hyperparameters and their possible values and distributions are described
in the `ConvNet.hyperparameters` method.

Implement the search using reinforcement learning. Notably, generate the
hyperparameters using a forward LSTM with dimensionality 16, generating
individual hyperparameters on each time step.

This task is evaluated manually. After you submit your solution to ReCodEx
(which will not pass automatic evaluation), write me an email and I will
perform the evaluation.
