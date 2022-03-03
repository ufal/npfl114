### Assignment: mnist_regularization
#### Date: Deadline: Mar 14, 7:59 a.m.
#### Points: 3 points
#### Tests: mnist_regularization_tests

You will learn how to implement three regularization methods in this assignment.
Start with the
[mnist_regularization.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_regularization.py)
template and implement the following:
- Allow using dropout with rate `args.dropout`. Add a dropout layer after the
  first `Flatten` and also after all `Dense` hidden layers (but not after the
  output layer).
- Allow using L2 regularization with weight `args.l2`. Use
  `tf.keras.regularizers.L2` as a regularizer for all kernels (but not
  biases) of all `Dense` layers (including the last one).
- Allow using label smoothing with weight `args.label_smoothing`. Instead
  of `SparseCategoricalCrossentropy`, you will need to use
  `CategoricalCrossentropy` which offers `label_smoothing` argument.

In ReCodEx, there will be six tests (two for each regularization methods) and
you will get half a point for passing each one.

In addition to submitting the task in ReCodEx, also run the following
variations and observe the results in TensorBoard
(or [online here](https://tensorboard.dev/experiment/9lu5xlnvTYODHPs2UkS5Jw/)),
notably the training, development and test set accuracy and loss:
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`;
- l2 regularization `0`, `0.001`, `0.0001`, `0.00001`;
- label smoothing `0`, `0.1`, `0.3`, `0.5`.

#### Tests Start: mnist_regularization_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_regularization.py --epochs=1 --dropout=0.3`
```
loss: 0.7987 - accuracy: 0.7616 - val_loss: 0.3175 - val_accuracy: 0.9110 - val_test_loss: 0.3825 - val_test_accuracy: 0.8882
```
- `python3 mnist_regularization.py --epochs=1 --dropout=0.5 --hidden_layers 300 300`
```
loss: 1.4363 - accuracy: 0.5090 - val_loss: 0.4447 - val_accuracy: 0.8862 - val_test_loss: 0.5256 - val_test_accuracy: 0.8537
```
- `python3 mnist_regularization.py --epochs=1 --l2=0.001`
```
loss: 0.9748 - accuracy: 0.8374 - val_loss: 0.5730 - val_accuracy: 0.9188 - val_test_loss: 0.6294 - val_test_accuracy: 0.9049
```
- `python3 mnist_regularization.py --epochs=1 --l2=0.0001`
```
loss: 0.6501 - accuracy: 0.8396 - val_loss: 0.3136 - val_accuracy: 0.9210 - val_test_loss: 0.3704 - val_test_accuracy: 0.9075
```
- `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.1`
```
loss: 0.9918 - accuracy: 0.8436 - val_loss: 0.7645 - val_accuracy: 0.9254 - val_test_loss: 0.8047 - val_test_accuracy: 0.9095
```
- `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.3`
```
loss: 1.5068 - accuracy: 0.8428 - val_loss: 1.3686 - val_accuracy: 0.9332 - val_test_loss: 1.3936 - val_test_accuracy: 0.9125
```
#### Tests End:
