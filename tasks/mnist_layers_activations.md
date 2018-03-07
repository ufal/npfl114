The motivation of this exercise is to famirialize a bit with TensorFlow and
TensorBoard. Start by playing with
[mnist_example.py](https://github.com/ufal/npfl114/tree/master/labs/01/mnist_example.py).
Run it, and when it finishes, run TensorBoard using `tensorboard --logdir
logs`. Then open <http://localhost:6006> in a browser and explore the three
active tabs – Scalars, Images and Graphs.

Your goal is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl114/tree/master/labs/01/mnist_layers_activations.py)
template and implement the following:
- A number of hidden layers (including zero) can be specified on the command line
  using parameter `layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.
- Print the final accuracy on the test set to standard output. Write the
  accuracy as percentage rounded on two decimal places, e.g., `91.23`.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard:
- 0 layers, activation none
- 1 layer, activation none, relu, tanh, sigmoid
- 3 layers, activation sigmoid, relu
- 5 layers, activation sigmoid
