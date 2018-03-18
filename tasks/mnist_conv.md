In this assignment, you will be training convolutional networks. Start with
the [mnist_conv.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_conv.py)
template and implement the following functionality. The architecture of the
network is described by the `cnn` parameter, which contains comma-separated
specifications of sequential layers:
- `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
  activation and specified number of filters, kernel size, stride and padding.
  Example: `C-10-3-1-same`
- `A-kernel_size-stride`: Add average pooling with specified size and stride.
  Example: `A-3-2`
- `F`: Flatten inputs.
- `R-hidden_layer_size`: Add a dense layer with ReLU activation and specified
  size. Example: `R-100`

For example, when using `--cnn=C-10-3-2-same,A-3-2,F,R-100`, the development
accuracies after first five epochs should be 92.54, 94.42, 95.40, 96.48 and 96.90.

After implementing this task, you should continue with `mnist_batchnorm`.
