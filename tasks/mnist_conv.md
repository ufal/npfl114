In this assignment, you will be training convolutional networks. Start with
the [mnist_conv.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_conv.py)
template and implement the following functionality using the `tf.layers` module.
The architecture of the
network is described by the `cnn` parameter, which contains comma-separated
specifications of sequential layers:
- `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
  activation and specified number of filters, kernel size, stride and padding.
  Example: `C-10-3-1-same`
- `M-kernel_size-stride`: Add max pooling with specified size and stride.
  Example: `M-3-2`
- `F`: Flatten inputs.
- `R-hidden_layer_size`: Add a dense layer with ReLU activation and specified
  size. Example: `R-100`

For example, when using `--cnn=C-10-3-2-same,M-3-2,F,R-100`, the development
accuracies after first five epochs on my CPU TensorFlow version are
95.14, 97.00, 97.68, 97.66, and 97.98. However, some students also obtained
slightly different results on their computers and still passed ReCodEx
evaluation.

After implementing this task, you should continue with `mnist_batchnorm`.
