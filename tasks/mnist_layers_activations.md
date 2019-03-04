### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 17, 23:59
#### Points: 3 points

In order to familiarize with TensorFlow and TensorBoard, start by playing with
[example_keras_tensorboard.py](https://github.com/ufal/npfl114/tree/master/labs/01/example_keras_tensorboard.py).
Run it, and when it finishes, run TensorBoard using `tensorboard --logdir logs`.
Then open <http://localhost:6006> in a browser and explore the active tabs.

Your goal is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl114/tree/master/labs/01/mnist_layers_activations.py)
template and implement the following:
- A number of hidden layers (including zero) can be specified on the command line
  using parameter `layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.
- Print the final accuracy on the test set.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard:
- 0 layers, activation none
- 1 layer, activation none, relu, tanh, sigmoid
- 10 layers, activation sigmoid, relu
