### Assignment: sgd_backpropagation
#### Date: Deadline: Mar 07, 7:59 a.m.
#### Points: 3 points
#### Examples: sgd_backpropagation_examples

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

#### Examples Start: sgd_backpropagation_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 sgd_backpropagation.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.84
Dev accuracy after epoch 2 is 93.86
Dev accuracy after epoch 3 is 94.64
Dev accuracy after epoch 4 is 95.24
Dev accuracy after epoch 5 is 95.26
Test accuracy after epoch 5 is 94.60
```
- `python3 sgd_backpropagation.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.66
Dev accuracy after epoch 2 is 95.00
Dev accuracy after epoch 3 is 95.72
Dev accuracy after epoch 4 is 95.80
Dev accuracy after epoch 5 is 96.34
Test accuracy after epoch 5 is 95.31
```
#### Examples End:
