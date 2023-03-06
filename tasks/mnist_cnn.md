### Assignment: mnist_cnn
#### Date: Deadline: Mar 20, 7:59 a.m.
#### Points: 3 points
#### Tests: mnist_cnn_tests

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

#### Tests Start: mnist_cnn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100`
```
loss: 0.3093 - accuracy: 0.9130 - val_loss: 0.1374 - val_accuracy: 0.9624
```
- `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100,D-0.5`
```
loss: 0.4770 - accuracy: 0.8594 - val_loss: 0.1624 - val_accuracy: 0.9552
```
- `python3 mnist_cnn.py --epochs=1 --cnn=M-5-2,F,H-50`
```
loss: 0.7365 - accuracy: 0.7773 - val_loss: 0.3899 - val_accuracy: 0.8800
```
- `python3 mnist_cnn.py --epochs=1 --cnn=C-8-3-5-same,C-8-3-2-valid,F,H-50`
```
loss: 0.8051 - accuracy: 0.7453 - val_loss: 0.3693 - val_accuracy: 0.8868
```
- `python3 mnist_cnn.py --epochs=1 --cnn=CB-6-3-5-valid,F,H-32`
```
loss: 0.5878 - accuracy: 0.8189 - val_loss: 0.2638 - val_accuracy: 0.9246
```
- `python3 mnist_cnn.py --epochs=1 --cnn=CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50`
```
loss: 0.4186 - accuracy: 0.8674 - val_loss: 0.1729 - val_accuracy: 0.9456
```
#### Tests End:
