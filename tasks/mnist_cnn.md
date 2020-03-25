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
_Note that the results might be slightly different, depending on whether you use
GPU or on your CPU type._

- `python3 mnist_cnn.py --batch_size=50 --cnn=F --epochs=3 --seed=42 --threads=1`
```
Epoch 1/3 - train_accuracy: 0.8625 - val_accuracy: 0.9288
Epoch 2/3 - train_accuracy: 0.9094 - val_accuracy: 0.9330
Epoch 3/3 - train_accuracy: 0.9164 - val_accuracy: 0.9386
Test accuracy: 0.9210
```
- `python3 mnist_cnn.py --batch_size=50 --cnn=F,H-100 --epochs=3 --seed=42 --threads=1`
```
Epoch 1/3 - train_accuracy: 0.9109 - val_accuracy: 0.9612
Epoch 2/3 - train_accuracy: 0.9583 - val_accuracy: 0.9704
Epoch 3/3 - train_accuracy: 0.9698 - val_accuracy: 0.9730
Test accuracy: 0.9698
```
- `python3 mnist_cnn.py --batch_size=50 --cnn=F,H-200,D-0.5 --epochs=3 --seed=42 --threads=1`
```
Epoch 1/3 - train_accuracy: 0.8899 - val_accuracy: 0.9658
Epoch 2/3 - train_accuracy: 0.9426 - val_accuracy: 0.9716
Epoch 3/3 - train_accuracy: 0.9518 - val_accuracy: 0.9758
Test accuracy: 0.9716
```
- `python3 mnist_cnn.py --batch_size=50 --cnn=C-16-3-2-same,M-3-2,F,H-100 --epochs=3 --seed=42 --threads=1`
```
Epoch 1/3 - train_accuracy: 0.9104 - val_accuracy: 0.9702
Epoch 2/3 - train_accuracy: 0.9674 - val_accuracy: 0.9754
Epoch 3/3 - train_accuracy: 0.9749 - val_accuracy: 0.9814
Test accuracy: 0.9772
```
- `python3 mnist_cnn.py --batch_size=50 --cnn=CB-16-3-2-same,M-3-2,F,H-100 --epochs=3 --seed=42 --threads=1`
```
Epoch 1/3 - train_accuracy: 0.9179 - val_accuracy: 0.9744
Epoch 2/3 - train_accuracy: 0.9704 - val_accuracy: 0.9792
Epoch 3/3 - train_accuracy: 0.9776 - val_accuracy: 0.9804
Test accuracy: 0.9770
```
- `python3 mnist_cnn.py --batch_size=50 --cnn=C-16-3-2-same,R-[C-16-3-1-same,C-16-3-1-same],M-3-2,F,H-100 --epochs=3 --seed=42 --threads=1`
```
Epoch 1/3 - train_accuracy: 0.9387 - val_accuracy: 0.9798
Epoch 2/3 - train_accuracy: 0.9804 - val_accuracy: 0.9890
Epoch 3/3 - train_accuracy: 0.9855 - val_accuracy: 0.9886
Test accuracy: 0.9870
```
#### Examples End:
