### Assignment: mnist_layers_activations
#### Date: Deadline: Feb 27, 7:59 a.m.
#### Points: 2 points
#### Tests: mnist_layers_activations_tests
#### Examples: mnist_layers_activations_examples

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

#### Tests Start: mnist_layers_activations_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=0 --activation=none`
```
loss: 0.5390 - accuracy: 0.8607 - val_loss: 0.2745 - val_accuracy: 0.9288
```
2. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=none`
```
loss: 0.3809 - accuracy: 0.8915 - val_loss: 0.2403 - val_accuracy: 0.9344
```
3. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=relu`
```
loss: 0.3093 - accuracy: 0.9130 - val_loss: 0.1374 - val_accuracy: 0.9624
```
4. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=tanh`
```
loss: 0.3304 - accuracy: 0.9067 - val_loss: 0.1601 - val_accuracy: 0.9580
```
5. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=sigmoid`
```
loss: 0.4905 - accuracy: 0.8771 - val_loss: 0.2123 - val_accuracy: 0.9452
```
6. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=3 --activation=relu`
```
loss: 0.2727 - accuracy: 0.9185 - val_loss: 0.1180 - val_accuracy: 0.9644
```
#### Tests End:
#### Examples Start: mnist_layers_activations_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_layers_activations.py --hidden_layers=0 --activation=none`
```
Epoch  1/10 loss: 0.5390 - accuracy: 0.8607 - val_loss: 0.2745 - val_accuracy: 0.9288
Epoch  5/10 loss: 0.2777 - accuracy: 0.9228 - val_loss: 0.2195 - val_accuracy: 0.9432
Epoch 10/10 loss: 0.2592 - accuracy: 0.9279 - val_loss: 0.2141 - val_accuracy: 0.9434
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=none`
```
Epoch  1/10 loss: 0.3809 - accuracy: 0.8915 - val_loss: 0.2403 - val_accuracy: 0.9344
Epoch  5/10 loss: 0.2759 - accuracy: 0.9220 - val_loss: 0.2338 - val_accuracy: 0.9348
Epoch 10/10 loss: 0.2642 - accuracy: 0.9257 - val_loss: 0.2322 - val_accuracy: 0.9386
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=relu`
```
Epoch  1/10 loss: 0.3093 - accuracy: 0.9130 - val_loss: 0.1374 - val_accuracy: 0.9624
Epoch  5/10 loss: 0.0613 - accuracy: 0.9809 - val_loss: 0.0733 - val_accuracy: 0.9798
Epoch 10/10 loss: 0.0226 - accuracy: 0.9934 - val_loss: 0.0751 - val_accuracy: 0.9784
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=tanh`
```
Epoch  1/10 loss: 0.3304 - accuracy: 0.9067 - val_loss: 0.1601 - val_accuracy: 0.9580
Epoch  5/10 loss: 0.0745 - accuracy: 0.9785 - val_loss: 0.0804 - val_accuracy: 0.9758
Epoch 10/10 loss: 0.0272 - accuracy: 0.9930 - val_loss: 0.0719 - val_accuracy: 0.9782
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=sigmoid`
```
Epoch  1/10 loss: 0.4905 - accuracy: 0.8771 - val_loss: 0.2123 - val_accuracy: 0.9452
Epoch  5/10 loss: 0.1228 - accuracy: 0.9647 - val_loss: 0.1037 - val_accuracy: 0.9708
Epoch 10/10 loss: 0.0604 - accuracy: 0.9834 - val_loss: 0.0790 - val_accuracy: 0.9754
```
- `python3 mnist_layers_activations.py --hidden_layers=3 --activation=relu`
```
Epoch  1/10 loss: 0.2727 - accuracy: 0.9185 - val_loss: 0.1180 - val_accuracy: 0.9644
Epoch  5/10 loss: 0.0501 - accuracy: 0.9837 - val_loss: 0.0944 - val_accuracy: 0.9734
Epoch 10/10 loss: 0.0242 - accuracy: 0.9919 - val_loss: 0.0936 - val_accuracy: 0.9814
```
- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=relu`
```
Epoch  1/10 loss: 0.3648 - accuracy: 0.8872 - val_loss: 0.1340 - val_accuracy: 0.9642
Epoch  5/10 loss: 0.0820 - accuracy: 0.9774 - val_loss: 0.0925 - val_accuracy: 0.9750
Epoch 10/10 loss: 0.0510 - accuracy: 0.9857 - val_loss: 0.0914 - val_accuracy: 0.9796
```
- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=sigmoid`
```
Epoch  1/10 loss: 2.2465 - accuracy: 0.1236 - val_loss: 1.9748 - val_accuracy: 0.1996
Epoch  5/10 loss: 0.5975 - accuracy: 0.8113 - val_loss: 0.4746 - val_accuracy: 0.8552
Epoch 10/10 loss: 0.3410 - accuracy: 0.9216 - val_loss: 0.3415 - val_accuracy: 0.9198
```
#### Examples End:
