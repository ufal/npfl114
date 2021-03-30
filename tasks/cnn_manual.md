### Assignment: cnn_manual
#### Date: Deadline: Apr 12, 23:59
#### Points: 3 points
#### Examples: cnn_manual_examples

To pass this assignment, you need to manually implement the forward and backward
pass through a 2D convolutional layer. Start with the
[cnn_manual.py](https://github.com/ufal/npfl114/tree/master/labs/05/cnn_manual.py)
template, which construct a series of 2D convolutional layers with ReLU
activation and `valid` padding, specified in the `args.cnn` option.
The `args.cnn` contains comma separater layer specifications in the format
`filters-kernel_size-stride`.

Of course, you cannot use any TensorFlow convolutional operation (instead,
implement the forward and backward pass using matrix multiplication and other
operations) nor the `GradientTape` for gradient computation.

#### Examples Start: cnn_manual_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 cnn_manual.py --cnn=5-1-1`
```
Dev accuracy after epoch 1 is 91.42
Dev accuracy after epoch 2 is 92.44
Dev accuracy after epoch 3 is 91.82
Dev accuracy after epoch 4 is 92.62
Dev accuracy after epoch 5 is 92.32
Test accuracy after epoch 5 is 90.73
```
- `python3 cnn_manual.py --cnn=5-3-1`
```
Dev accuracy after epoch 1 is 95.62
Dev accuracy after epoch 2 is 96.06
Dev accuracy after epoch 3 is 96.22
Dev accuracy after epoch 4 is 96.46
Dev accuracy after epoch 5 is 96.12
Test accuracy after epoch 5 is 95.73
```
- `python3 cnn_manual.py --cnn=5-3-2`
```
Dev accuracy after epoch 1 is 93.14
Dev accuracy after epoch 2 is 94.90
Dev accuracy after epoch 3 is 95.26
Dev accuracy after epoch 4 is 95.42
Dev accuracy after epoch 5 is 95.34
Test accuracy after epoch 5 is 95.01
```
- `python3 cnn_manual.py --cnn=5-3-2,10-3-2`
```
Dev accuracy after epoch 1 is 95.00
Dev accuracy after epoch 2 is 96.40
Dev accuracy after epoch 3 is 96.42
Dev accuracy after epoch 4 is 96.84
Dev accuracy after epoch 5 is 97.16
Test accuracy after epoch 5 is 96.44
```
#### Examples End:
