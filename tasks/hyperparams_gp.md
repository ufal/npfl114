The goal of this assignment is to try performing automatic hyperparameter
search. Your goal is to optimize [conv_net.py](https://github.com/ufal/npfl114/tree/master/labs/13/conv_net.py)
model with several hyperparameters, so that it achieves highest validation
accuracy on Fashion MNIST dataset after two epochs of training.
The hyperparameters and their possible values and distributions are described
in the `ConvNet.hyperparameters` method.

Implement the search using the `skopt` package (can be installed using
`pip3 install [--user] scikit-optimize`), and print best accuracy
after 15 trials. Implement the two following strategies:
- `random_search`: use random search in the hyperparameter space
- `gp_ei`: use gaussian process approach (`skopt.gp_minimize`) with
  _expected improvement (EI)_ acquisition function

This task is evaluated manually. After you submit your solution to ReCodEx
(which will not pass automatic evaluation), write me an email and I will
perform the evaluation.
