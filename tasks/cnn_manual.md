### Assignment: cnn_manual
#### Date: Deadline: Apr 19, 23:59
#### Points: 3 points
#### Examples: cnn_manual_example

To pass this assignment, you need to manually implement the forward and backward
pass through a 2D convolutional layer. Start with the
[cnn_manual.py](https://github.com/ufal/npfl114/tree/master/labs/06/cnn_manual.py)
template, which construct a series of 2D convolutional layers with ReLU
activation and `valid` padding, specified in the `args.cnn` option.
The `args.cnn` contains comma separater layer specifications in the format
`filters-kernel_size-stride`.

Of course, you cannot use any TensorFlow convolutional operation (instead,
implement the forward and backward pass using matrix multiplication and other
operations) nor the `GradientTape` for gradient computation.

#### Examples Start: cnn_manual_example
_Note that the results might be slightly different, depending on whether you use
GPU or on your CPU type._

- `python3 cnn_manual.py --batch_size=64 --cnn=4-1-1 --epochs=3 --learning_rate=0.01 --seed=42 --threads=1`
```
Dev accuracy after epoch 1 is 91.54
Dev accuracy after epoch 2 is 92.04
Dev accuracy after epoch 3 is 92.00
Test accuracy after epoch 3 is 89.70
```
- `python3 cnn_manual.py --batch_size=64 --cnn=4-3-1 --epochs=3 --learning_rate=0.01 --seed=42 --threads=1`
```
Dev accuracy after epoch 1 is 93.24
Dev accuracy after epoch 2 is 93.04
Dev accuracy after epoch 3 is 95.16
Test accuracy after epoch 3 is 93.77
```
- `python3 cnn_manual.py --batch_size=64 --cnn=4-3-2 --epochs=3 --learning_rate=0.01 --seed=42 --threads=1`
```
Dev accuracy after epoch 1 is 92.52
Dev accuracy after epoch 2 is 92.62
Dev accuracy after epoch 3 is 94.14
Test accuracy after epoch 3 is 92.42
```
- `python3 cnn_manual.py --batch_size=64 --cnn=4-3-2,8-3-2 --epochs=3 --learning_rate=0.01 --seed=42 --threads=1`
```
Dev accuracy after epoch 1 is 93.18
Dev accuracy after epoch 2 is 94.22
Dev accuracy after epoch 3 is 95.52
Test accuracy after epoch 3 is 93.83
```
#### Examples End:
