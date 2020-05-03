### Assignment: mnist_cnn
#### Date: Deadline: Apr 05, 23:59
#### Points: 5 points
#### Examples: mnist_cnn_example

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
- `M-kernel_size-stride`: Add max pooling with specified size and stride.
  Example: `M-3-2`
- `R-[layers]`: Add a residual connection. The `layers` contain a specification
  of at least one convolutional layer (but not a recursive residual connection `R`).
  The input to the specified layers is then added to their output (after the
  ReLU nonlinearity of the last one).
  Example: `R-[C-16-3-1-same,C-16-3-1-same]`
- `F`: Flatten inputs. Must appear exactly once in the architecture.
- `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified
  size. Example: `H-100`
- `D-dropout_rate`: Apply dropout with the given dropout rate. Example: `D-0.5`

An example architecture might be `--cnn=CB-16-5-2-same,M-3-2,F,H-100,D-0.5`.
You can assume the resulting network is valid; it is fine to crash if it is not.

After a successful ReCodEx submission, you can try obtaining the best accuracy
on MNIST and then advance to `cifar_competition`.

#### Examples Start: mnist_cnn_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 mnist_cnn.py --seed=7 --recodex --threads=1 --epochs=1 --batch_size=50 --cnn=F,H-100`
  ```
  94.84
  ```
- `python3 mnist_cnn.py --seed=7 --recodex --threads=1 --epochs=1 --batch_size=50 --cnn=F,H-100,D-0.5`
  ```
  94.17
  ```
- `python3 mnist_cnn.py --seed=7 --recodex --threads=1 --epochs=1 --batch_size=50 --cnn=M-5-2,F,H-50`
  ```
  87.18
  ```
- `python3 mnist_cnn.py --seed=7 --recodex --threads=1 --epochs=1 --batch_size=50 --cnn=C-8-3-5-same,C-8-3-2-valid,F,H-50`
  ```
  86.18
  ```
- `python3 mnist_cnn.py --seed=7 --recodex --threads=1 --epochs=1 --batch_size=50 --cnn=CB-6-3-5-valid,F,H-32`
  ```
  90.23
  ```
- `python3 mnist_cnn.py --seed=7 --recodex --threads=1 --epochs=1 --batch_size=50 --cnn=C-8-3-5-valid,R-[C-8-3-1-same,C-8-3-1-same],F,H-50`
  ```
  91.15
  ```
#### Examples End:
