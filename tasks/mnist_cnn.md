### Assignment: mnist_cnn
#### Date: Deadline: Apr 05, 23:59
#### Points: 4 points
#### Examples: mnist_cnn_examples

To pass this assignment, you will learn to construct basic convolutional
neural network layers. Start with the
[mnist_cnn.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_cnn.py)
template and assume the requested architecture is described by the `cnn`
argument, which contains comma-separated specifications of the following layers:
- `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
  activation and specified number of filters, kernel size, stride and padding.
  Example: `C-10-3-1-same`
- `CB-filters-kernel_size-stride-padding`: Same as
  `C-filters-kernel_size-stride-padding`, but use batch normalization.
  In detail, start with a convolutional layer **without bias and activation**,
  then add batch normalization layer, and finally ReLU activation.
  Example: `CB-10-3-1-same`
- `M-pool_size-stride`: Add max pooling with specified size and stride, using
  the default `"valid"` padding.
  Example: `M-3-2`
- `R-[layers]`: Add a residual connection. The `layers` contain a specification
  of at least one convolutional layer (but not a recursive residual connection `R`).
  The input to the `R` layer should be processed sequentially by `layers`, and the
  produced output (after the ReLU nonlinearty of the last layer) should be added
  to the input (of this `R` layer).
  Example: `R-[C-16-3-1-same,C-16-3-1-same]`
- `F`: Flatten inputs. Must appear exactly once in the architecture.
- `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified
  size. Example: `H-100`
- `D-dropout_rate`: Apply dropout with the given dropout rate. Example: `D-0.5`

An example architecture might be `--cnn=CB-16-5-2-same,M-3-2,F,H-100,D-0.5`.
You can assume the resulting network is valid; it is fine to crash if it is not.

After a successful ReCodEx submission, you can try obtaining the best accuracy
on MNIST and then advance to `cifar_competition`.

#### Examples Start: mnist_cnn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 mnist_cnn.py --cnn=F,H-100`
```
Epoch 1/5 loss: 0.5379 - accuracy: 0.8500 - val_loss: 0.1459 - val_accuracy: 0.9612
Epoch 2/5 loss: 0.1563 - accuracy: 0.9553 - val_loss: 0.1128 - val_accuracy: 0.9682
Epoch 3/5 loss: 0.1052 - accuracy: 0.9697 - val_loss: 0.0966 - val_accuracy: 0.9714
Epoch 4/5 loss: 0.0792 - accuracy: 0.9765 - val_loss: 0.0864 - val_accuracy: 0.9744
Epoch 5/5 loss: 0.0627 - accuracy: 0.9814 - val_loss: 0.0818 - val_accuracy: 0.9768
loss: 0.0844 - accuracy: 0.9757
```
- `python3 mnist_cnn.py --cnn=F,H-100,D-0.5`
```
Epoch 1/5 loss: 0.7447 - accuracy: 0.7719 - val_loss: 0.1617 - val_accuracy: 0.9596
Epoch 2/5 loss: 0.2781 - accuracy: 0.9167 - val_loss: 0.1266 - val_accuracy: 0.9668
Epoch 3/5 loss: 0.2293 - accuracy: 0.9321 - val_loss: 0.1097 - val_accuracy: 0.9696
Epoch 4/5 loss: 0.2003 - accuracy: 0.9399 - val_loss: 0.1035 - val_accuracy: 0.9716
Epoch 5/5 loss: 0.1858 - accuracy: 0.9444 - val_loss: 0.1019 - val_accuracy: 0.9728
loss: 0.1131 - accuracy: 0.9676
```
- `python3 mnist_cnn.py --cnn=M-5-2,F,H-50`
```
Epoch 1/5 loss: 1.0752 - accuracy: 0.6618 - val_loss: 0.3934 - val_accuracy: 0.8818
Epoch 2/5 loss: 0.4421 - accuracy: 0.8598 - val_loss: 0.3241 - val_accuracy: 0.9000
Epoch 3/5 loss: 0.3651 - accuracy: 0.8849 - val_loss: 0.2996 - val_accuracy: 0.9078
Epoch 4/5 loss: 0.3271 - accuracy: 0.8951 - val_loss: 0.2712 - val_accuracy: 0.9174
Epoch 5/5 loss: 0.3014 - accuracy: 0.9049 - val_loss: 0.2632 - val_accuracy: 0.9182
loss: 0.2967 - accuracy: 0.9067
```
- `python3 mnist_cnn.py --cnn=C-8-3-5-same,C-8-3-2-valid,F,H-50`
```
Epoch 1/5 loss: 1.1907 - accuracy: 0.6001 - val_loss: 0.3445 - val_accuracy: 0.9004
Epoch 2/5 loss: 0.4124 - accuracy: 0.8730 - val_loss: 0.2818 - val_accuracy: 0.9158
Epoch 3/5 loss: 0.3335 - accuracy: 0.8970 - val_loss: 0.2523 - val_accuracy: 0.9254
Epoch 4/5 loss: 0.3036 - accuracy: 0.9043 - val_loss: 0.2292 - val_accuracy: 0.9316
Epoch 5/5 loss: 0.2802 - accuracy: 0.9143 - val_loss: 0.2186 - val_accuracy: 0.9340
loss: 0.2520 - accuracy: 0.9243
```
- `python3 mnist_cnn.py --cnn=CB-6-3-5-valid,F,H-32`
```
Epoch 1/5 loss: 0.9799 - accuracy: 0.6768 - val_loss: 0.2519 - val_accuracy: 0.9230
Epoch 2/5 loss: 0.3122 - accuracy: 0.9045 - val_loss: 0.2116 - val_accuracy: 0.9338
Epoch 3/5 loss: 0.2493 - accuracy: 0.9230 - val_loss: 0.1792 - val_accuracy: 0.9496
Epoch 4/5 loss: 0.2147 - accuracy: 0.9322 - val_loss: 0.1637 - val_accuracy: 0.9528
Epoch 5/5 loss: 0.1873 - accuracy: 0.9415 - val_loss: 0.1544 - val_accuracy: 0.9566
loss: 0.1857 - accuracy: 0.9424
```
- `python3 mnist_cnn.py --cnn=CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50`
```
Epoch 1/5 loss: 0.7976 - accuracy: 0.7449 - val_loss: 0.1791 - val_accuracy: 0.9458
Epoch 2/5 loss: 0.2052 - accuracy: 0.9360 - val_loss: 0.1531 - val_accuracy: 0.9506
Epoch 3/5 loss: 0.1497 - accuracy: 0.9524 - val_loss: 0.1340 - val_accuracy: 0.9600
Epoch 4/5 loss: 0.1261 - accuracy: 0.9593 - val_loss: 0.1226 - val_accuracy: 0.9624
Epoch 5/5 loss: 0.1113 - accuracy: 0.9642 - val_loss: 0.1094 - val_accuracy: 0.9684
loss: 0.1212 - accuracy: 0.9609
```
#### Examples End:
