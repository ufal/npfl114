### Assignment: cnn_manual
#### Date: Deadline: Mar 28, 7:59
#### Points: 3 points

To pass this assignment, you need to manually implement the forward and backward
pass through a 2D convolutional layer. Start with the
(TEMPLATE TO APPEAR SOON), which construct a series of 2D convolutional layers with ReLU
activation and `valid` padding, specified in the `args.cnn` option.
The `args.cnn` contains comma separater layer specifications in the format
`filters-kernel_size-stride`.

Of course, you cannot use any TensorFlow convolutional operation (instead,
implement the forward and backward pass using matrix multiplication and other
operations) nor the `GradientTape` for gradient computation.
