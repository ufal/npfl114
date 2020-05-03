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
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 cnn_manual.py --seed=7 --recodex --threads=1 --learning_rate=0.01 --epochs=1 --batch_size=50 --cnn=5-1-1`
  ```
  89.62
  ```
- `python3 cnn_manual.py --seed=7 --recodex --threads=1 --learning_rate=0.01 --epochs=1 --batch_size=50 --cnn=5-3-1`
  ```
  92.83
  ```
- `python3 cnn_manual.py --seed=7 --recodex --threads=1 --learning_rate=0.01 --epochs=1 --batch_size=50 --cnn=5-3-2`
  ```
  90.62
  ```
- `python3 cnn_manual.py --seed=7 --recodex --threads=1 --learning_rate=0.01 --epochs=1 --batch_size=50 --cnn=5-3-2,10-3-2`
  ```
  92.58
  ```
#### Examples End:
