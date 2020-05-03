### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 8, 23:59
#### Points: 2 points
#### Examples: mnist_layers_activations_example

Before solving the assignment, start by playing with
[example_keras_tensorboard.py](https://github.com/ufal/npfl114/tree/master/labs/01/example_keras_tensorboard.py),
in order to familiarize with TensorFlow and TensorBoard.
Run it, and when it finishes, run TensorBoard using `tensorboard --logdir logs`.
Then open <http://localhost:6006> in a browser and explore the active tabs.

**Your goal** is to modify the
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
- `0` layers, activation `none`
- `1` layer, activation `none`, `relu`, `tanh`, `sigmoid`
- `10` layers, activation `sigmoid`, `relu`

#### Examples Start: mnist_layers_activations_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=0 --activation=none`
  ```
  91.22
  ```
- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=1 --activation=none`
  ```
  91.96
  ```
- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=1 --activation=relu`
  ```
  94.84
  ```
- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=1 --activation=tanh`
  ```
  94.19
  ```
- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=1 --activation=sigmoid`
  ```
  92.32
  ```
- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=3 --activation=relu`
  ```
  96.06
  ```
- `python3 mnist_layers_activations.py --recodex --seed=7 --threads=1 --epochs=1 --batch_size=50 --layers=5 --activation=tanh`
  ```
  94.67
  ```
#### Examples End:
