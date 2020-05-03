### Assignment: sgd_backpropagation
#### Date: Deadline: Mar ~~15~~ 22, 23:59
#### Points: 3 points
#### Examples: sgd_backpropagation_example

In this exercise you will learn how to compute gradients using the so-called
**automatic differentiation**, which is implemented by an automated
backpropagation algorithm in TensorFlow. You will then perform training
by running manually implemented minibatch stochastic gradient descent.

Starting with the
[sgd_backpropagation.py](https://github.com/ufal/npfl114/tree/master/labs/02/sgd_backpropagation.py)
template, you should:
- implement a neural network with a single _tanh_ hidden layer and
  categorical output layer;
- compute the crossentropy loss;
- use `tf.GradientTape` to automatically compute the gradient of the loss
  with respect to all variables;
- perform the SGD update.

#### Examples Start: sgd_backpropagation_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 sgd_backpropagation.py --batch_size=64 --epochs=2 --hidden_layer=20 --learning_rate=0.1 --seed=7 --threads=1`
  ```
  92.38
  ```
- `python3 sgd_backpropagation.py --batch_size=100 --epochs=2 --hidden_layer=32 --learning_rate=0.2 --seed=7 --threads=1`
  ```
  93.77
  ```
#### Examples End:
