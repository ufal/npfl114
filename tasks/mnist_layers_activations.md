### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 07, 7:59 a.m.
#### Points: 2 points
#### Examples: mnist_layers_activations_examples
#### Tests: mnist_layers_activations_tests

Before solving the assignment, start by playing with
[example_keras_tensorboard.py](https://github.com/ufal/npfl114/tree/master/labs/01/example_keras_tensorboard.py),
in order to familiarize with TensorFlow and TensorBoard.
Run it, and when it finishes, run TensorBoard using `tensorboard --logdir logs`.
Then open <http://localhost:6006> in a browser and explore the active tabs.

**Your goal** is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl114/tree/master/labs/01/mnist_layers_activations.py)
template and implement the following:
- A number of hidden layers (including zero) can be specified on the command line
  using parameter `hidden_layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.
- Print the final accuracy on the test set.

#### Examples Start: mnist_layers_activations_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_layers_activations.py --hidden_layers=0 --activation=none`
```
Epoch  1/10 loss: 0.5383 - accuracy: 0.8613 - val_loss: 0.2755 - val_accuracy: 0.9308
Epoch  5/10 loss: 0.2783 - accuracy: 0.9220 - val_loss: 0.2202 - val_accuracy: 0.9430
Epoch 10/10 loss: 0.2595 - accuracy: 0.9273 - val_loss: 0.2146 - val_accuracy: 0.9434
loss: 0.2637 - accuracy: 0.9259
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=none`
```
Epoch  1/10 loss: 0.3828 - accuracy: 0.8914 - val_loss: 0.2438 - val_accuracy: 0.9350
Epoch  5/10 loss: 0.2754 - accuracy: 0.9222 - val_loss: 0.2341 - val_accuracy: 0.9370
Epoch 10/10 loss: 0.2640 - accuracy: 0.9260 - val_loss: 0.2318 - val_accuracy: 0.9400
loss: 0.2795 - accuracy: 0.9241
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=relu`
```
Epoch  1/10 loss: 0.3195 - accuracy: 0.9109 - val_loss: 0.1459 - val_accuracy: 0.9612
Epoch  5/10 loss: 0.0629 - accuracy: 0.9811 - val_loss: 0.0820 - val_accuracy: 0.9776
Epoch 10/10 loss: 0.0237 - accuracy: 0.9937 - val_loss: 0.0801 - val_accuracy: 0.9776
loss: 0.0829 - accuracy: 0.9769
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=tanh`
```
Epoch  1/10 loss: 0.3414 - accuracy: 0.9039 - val_loss: 0.1668 - val_accuracy: 0.9570
Epoch  5/10 loss: 0.0750 - accuracy: 0.9783 - val_loss: 0.0813 - val_accuracy: 0.9774
Epoch 10/10 loss: 0.0268 - accuracy: 0.9937 - val_loss: 0.0788 - val_accuracy: 0.9744
loss: 0.0822 - accuracy: 0.9751
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=sigmoid`
```
Epoch  1/10 loss: 0.4969 - accuracy: 0.8751 - val_loss: 0.2150 - val_accuracy: 0.9400
Epoch  5/10 loss: 0.1222 - accuracy: 0.9649 - val_loss: 0.1041 - val_accuracy: 0.9718
Epoch 10/10 loss: 0.0594 - accuracy: 0.9842 - val_loss: 0.0805 - val_accuracy: 0.9772
loss: 0.0862 - accuracy: 0.9741
```
- `python3 mnist_layers_activations.py --hidden_layers=3 --activation=relu`
```
Epoch  1/10 loss: 0.2753 - accuracy: 0.9173 - val_loss: 0.1128 - val_accuracy: 0.9672
Epoch  5/10 loss: 0.0489 - accuracy: 0.9843 - val_loss: 0.0878 - val_accuracy: 0.9778
Epoch 10/10 loss: 0.0226 - accuracy: 0.9923 - val_loss: 0.0892 - val_accuracy: 0.9788
loss: 0.0770 - accuracy: 0.9793
```
- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=relu`
```
Epoch  1/10 loss: 0.3598 - accuracy: 0.8881 - val_loss: 0.1457 - val_accuracy: 0.9586
Epoch  5/10 loss: 0.0822 - accuracy: 0.9775 - val_loss: 0.1135 - val_accuracy: 0.9766
Epoch 10/10 loss: 0.0525 - accuracy: 0.9859 - val_loss: 0.1108 - val_accuracy: 0.9768
loss: 0.1342 - accuracy: 0.9715
```
- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=sigmoid`
```
Epoch  1/10 loss: 2.2830 - accuracy: 0.1088 - val_loss: 1.9021 - val_accuracy: 0.2120
Epoch  5/10 loss: 0.9505 - accuracy: 0.6286 - val_loss: 0.7622 - val_accuracy: 0.7214
Epoch 10/10 loss: 0.4468 - accuracy: 0.8919 - val_loss: 0.3524 - val_accuracy: 0.9212
loss: 0.4232 - accuracy: 0.8993
```
#### Examples End:
#### Tests Start: mnist_layers_activations_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=0 --activation=none`
```
Epoch  1/1 loss: 0.5383 - accuracy: 0.8613 - val_loss: 0.2755 - val_accuracy: 0.9308
loss: 0.3304 - accuracy: 0.9110
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=none`
```
Epoch 1/1 loss: 0.3828 - accuracy: 0.8914 - val_loss: 0.2438 - val_accuracy: 0.9350
loss: 0.2956 - accuracy: 0.9198
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=relu`
```
Epoch 1/1 loss: 0.3195 - accuracy: 0.9109 - val_loss: 0.1459 - val_accuracy: 0.9612
loss: 0.1738 - accuracy: 0.9517
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=tanh`
```
Epoch 1/1 loss: 0.3414 - accuracy: 0.9039 - val_loss: 0.1668 - val_accuracy: 0.9570
loss: 0.2039 - accuracy: 0.9422
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=sigmoid`
```
Epoch 1/1 loss: 0.4969 - accuracy: 0.8751 - val_loss: 0.2150 - val_accuracy: 0.9400
loss: 0.2627 - accuracy: 0.9268
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=3 --activation=relu`
```
Epoch 1/1 loss: 0.2753 - accuracy: 0.9173 - val_loss: 0.1128 - val_accuracy: 0.9672
loss: 0.1309 - accuracy: 0.9601
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=10 --activation=relu`
```
Epoch 1/1 loss: 0.3598 - accuracy: 0.8881 - val_loss: 0.1457 - val_accuracy: 0.9586
loss: 0.1806 - accuracy: 0.9474
```
- `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=10 --activation=sigmoid`
```
Epoch 1/1 loss: 2.2830 - accuracy: 0.1088 - val_loss: 1.9021 - val_accuracy: 0.2120
loss: 1.9469 - accuracy: 0.2065
```
#### Tests End:
