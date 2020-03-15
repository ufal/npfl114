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
_Note that the results can be different when a GPU is used, or when more than one
CPU thread is used. The results might be even different for a single-threaded CPU
(depending on CPU type), so do not hesitate to submit to ReCodEx when you have
similar but not exactly the same results._

Running
```
python3 sgd_backpropagation.py --batch_size=50 --epochs=5 --hidden_layer=100 --learning_rate=0.2 --seed=42 --threads=1
```
should give you
```
Dev accuracy after epoch 1 is 95.46
Dev accuracy after epoch 2 is 96.20
Dev accuracy after epoch 3 is 96.90
Dev accuracy after epoch 4 is 97.24
Dev accuracy after epoch 5 is 97.32
Test accuracy after epoch 5 is 96.99
```
#### Examples End:
