### Assignment: mnist_regularization
#### Date: Deadline: Mar 29, 23:59
#### Points: 3 points
#### Examples: mnist_regularization_examples

You will learn how to implement three regularization methods in this assignment.
Start with the
[mnist_regularization.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_regularization.py)
template and implement the following:
- Allow using dropout with rate `args.dropout`. Add a dropout layer after the
  first `Flatten` and also after all `Dense` hidden layers (but not after the
  output layer).
- Allow using L2 regularization with weight `args.l2`. Use
  `tf.keras.regularizers.L1L2` as a regularizer for all kernels (but not
  biases) of all `Dense` layers (including the last one).
- Allow using label smoothing with weight `args.label_smoothing`. Instead
  of `SparseCategoricalCrossentropy`, you will need to use
  `CategoricalCrossentropy` which offers `label_smoothing` argument.

In ReCodEx, there will be six tests tests (two for each regularization methods) and
you will get half a point for passing each one.

In addition to submitting the task in ReCodEx, also run the following
variations and observe the results in TensorBoard (notably training, development
and test set accuracy and loss):
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`;
- l2 regularization `0`, `0.001`, `0.0001`, `0.00001`;
- label smoothing `0`, `0.1`, `0.3`, `0.5`.

#### Examples Start: mnist_regularization_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 mnist_regularization.py --dropout=0.3`
```
Epoch  5/30 loss: 0.2319 - accuracy: 0.9309 - val_loss: 0.1919 - val_accuracy: 0.9420
Epoch 10/30 loss: 0.1207 - accuracy: 0.9608 - val_loss: 0.1507 - val_accuracy: 0.9560
Epoch 15/30 loss: 0.0785 - accuracy: 0.9758 - val_loss: 0.1300 - val_accuracy: 0.9606
Epoch 20/30 loss: 0.0595 - accuracy: 0.9833 - val_loss: 0.1292 - val_accuracy: 0.9628
Epoch 25/30 loss: 0.0517 - accuracy: 0.9816 - val_loss: 0.1311 - val_accuracy: 0.9618
Epoch 30/30 loss: 0.0315 - accuracy: 0.9919 - val_loss: 0.1413 - val_accuracy: 0.9618
loss: 0.1630 - accuracy: 0.9541
```
- `python3 mnist_regularization.py --dropout=0.5`
```
Epoch  5/30 loss: 0.3931 - accuracy: 0.8815 - val_loss: 0.2147 - val_accuracy: 0.9366
Epoch 10/30 loss: 0.2626 - accuracy: 0.9232 - val_loss: 0.1665 - val_accuracy: 0.9528
Epoch 15/30 loss: 0.2229 - accuracy: 0.9261 - val_loss: 0.1427 - val_accuracy: 0.9582
Epoch 20/30 loss: 0.1765 - accuracy: 0.9473 - val_loss: 0.1379 - val_accuracy: 0.9596
Epoch 25/30 loss: 0.1653 - accuracy: 0.9477 - val_loss: 0.1272 - val_accuracy: 0.9628
Epoch 30/30 loss: 0.1335 - accuracy: 0.9596 - val_loss: 0.1251 - val_accuracy: 0.9638
loss: 0.1510 - accuracy: 0.9521
```
- `python3 mnist_regularization.py --l2=0.001`
```
Epoch  5/30 loss: 0.3280 - accuracy: 0.9699 - val_loss: 0.3755 - val_accuracy: 0.9426
Epoch 10/30 loss: 0.2259 - accuracy: 0.9867 - val_loss: 0.3511 - val_accuracy: 0.9408
Epoch 15/30 loss: 0.2089 - accuracy: 0.9866 - val_loss: 0.3109 - val_accuracy: 0.9516
Epoch 20/30 loss: 0.1966 - accuracy: 0.9911 - val_loss: 0.2973 - val_accuracy: 0.9532
Epoch 25/30 loss: 0.1928 - accuracy: 0.9947 - val_loss: 0.3079 - val_accuracy: 0.9510
Epoch 30/30 loss: 0.1916 - accuracy: 0.9918 - val_loss: 0.3002 - val_accuracy: 0.9522
loss: 0.3313 - accuracy: 0.9394
```
- `python3 mnist_regularization.py --l2=0.0001`
```
Epoch  5/30 loss: 0.1387 - accuracy: 0.9793 - val_loss: 0.2231 - val_accuracy: 0.9452
Epoch 10/30 loss: 0.0686 - accuracy: 0.9982 - val_loss: 0.2132 - val_accuracy: 0.9508
Epoch 15/30 loss: 0.0530 - accuracy: 1.0000 - val_loss: 0.1938 - val_accuracy: 0.9564
Epoch 20/30 loss: 0.0446 - accuracy: 1.0000 - val_loss: 0.1954 - val_accuracy: 0.9538
Epoch 25/30 loss: 0.0431 - accuracy: 1.0000 - val_loss: 0.1909 - val_accuracy: 0.9572
Epoch 30/30 loss: 0.0439 - accuracy: 1.0000 - val_loss: 0.1914 - val_accuracy: 0.9608
loss: 0.2141 - accuracy: 0.9512
```
- `python3 mnist_regularization.py --label_smoothing=0.1`
```
Epoch  5/30 loss: 0.6077 - accuracy: 0.9865 - val_loss: 0.6626 - val_accuracy: 0.9610
Epoch 10/30 loss: 0.5422 - accuracy: 0.9994 - val_loss: 0.6414 - val_accuracy: 0.9642
Epoch 15/30 loss: 0.5225 - accuracy: 1.0000 - val_loss: 0.6324 - val_accuracy: 0.9654
Epoch 20/30 loss: 0.5145 - accuracy: 1.0000 - val_loss: 0.6289 - val_accuracy: 0.9674
Epoch 25/30 loss: 0.5101 - accuracy: 1.0000 - val_loss: 0.6281 - val_accuracy: 0.9678
Epoch 30/30 loss: 0.5081 - accuracy: 1.0000 - val_loss: 0.6271 - val_accuracy: 0.9682
loss: 0.6449 - accuracy: 0.9592
```
- `python3 mnist_regularization.py --label_smoothing=0.3`
```
Epoch  5/30 loss: 1.2506 - accuracy: 0.9884 - val_loss: 1.2963 - val_accuracy: 0.9630
Epoch 10/30 loss: 1.2070 - accuracy: 0.9992 - val_loss: 1.2799 - val_accuracy: 0.9652
Epoch 15/30 loss: 1.1937 - accuracy: 1.0000 - val_loss: 1.2773 - val_accuracy: 0.9638
Epoch 20/30 loss: 1.1875 - accuracy: 1.0000 - val_loss: 1.2748 - val_accuracy: 0.9662
Epoch 25/30 loss: 1.1847 - accuracy: 1.0000 - val_loss: 1.2753 - val_accuracy: 0.9676
Epoch 30/30 loss: 1.1834 - accuracy: 1.0000 - val_loss: 1.2760 - val_accuracy: 0.9660
loss: 1.2875 - accuracy: 0.9587
```
#### Examples End:
