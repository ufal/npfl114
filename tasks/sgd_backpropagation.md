### Assignment: sgd_backpropagation
#### Date: Deadline: Mar 07, 7:59 a.m.
#### Points: 3 points
#### Examples: sgd_backpropagation_examples
#### Tests: sgd_backpropagation_tests

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
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 sgd_backpropagation.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.84
Dev accuracy after epoch 2 is 93.86
Dev accuracy after epoch 3 is 94.64
Dev accuracy after epoch 4 is 95.24
Dev accuracy after epoch 5 is 95.26
Dev accuracy after epoch 6 is 95.66
Dev accuracy after epoch 7 is 95.58
Dev accuracy after epoch 8 is 95.86
Dev accuracy after epoch 9 is 96.18
Dev accuracy after epoch 10 is 96.08
Test accuracy after epoch 10 is 95.53
```
- `python3 sgd_backpropagation.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.66
Dev accuracy after epoch 2 is 95.00
Dev accuracy after epoch 3 is 95.72
Dev accuracy after epoch 4 is 95.80
Dev accuracy after epoch 5 is 96.34
Dev accuracy after epoch 6 is 96.16
Dev accuracy after epoch 7 is 96.42
Dev accuracy after epoch 8 is 96.36
Dev accuracy after epoch 9 is 96.60
Dev accuracy after epoch 10 is 96.58
Test accuracy after epoch 10 is 96.18
```
#### Examples End:
#### Tests Start: sgd_backpropagation_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 sgd_backpropagation.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.84
Dev accuracy after epoch 2 is 93.86
Test accuracy after epoch 2 is 93.21
```
- `python3 sgd_backpropagation.py --epochs=2 --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.66
Dev accuracy after epoch 2 is 95.00
Test accuracy after epoch 2 is 93.93
```
#### Tests End:
