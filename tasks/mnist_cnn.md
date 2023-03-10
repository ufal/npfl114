### Assignment: mnist_cnn
#### Date: Deadline: Mar 20, 7:59 a.m.
#### Points: 3 points
#### Tests: mnist_cnn_tests
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

#### Tests Start: mnist_cnn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100`
```
loss: 0.3093 - accuracy: 0.9130 - val_loss: 0.1374 - val_accuracy: 0.9624
```
2. `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100,D-0.5`
```
loss: 0.4770 - accuracy: 0.8594 - val_loss: 0.1624 - val_accuracy: 0.9552
```
3. `python3 mnist_cnn.py --epochs=1 --cnn=M-5-2,F,H-50`
```
loss: 0.7365 - accuracy: 0.7773 - val_loss: 0.3899 - val_accuracy: 0.8800
```
4. `python3 mnist_cnn.py --epochs=1 --cnn=C-8-3-5-same,C-8-3-2-valid,F,H-50`
```
loss: 0.8051 - accuracy: 0.7453 - val_loss: 0.3693 - val_accuracy: 0.8868
```
5. `python3 mnist_cnn.py --epochs=1 --cnn=CB-6-3-5-valid,F,H-32`
```
loss: 0.5878 - accuracy: 0.8189 - val_loss: 0.2638 - val_accuracy: 0.9246
```
6. `python3 mnist_cnn.py --epochs=1 --cnn=CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50`
```
loss: 0.4186 - accuracy: 0.8674 - val_loss: 0.1729 - val_accuracy: 0.9456
```
#### Tests End:
#### Examples Start: mnist_cnn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_cnn.py --cnn=F,H-100`
```
Epoch  1/10 loss: 0.3093 - accuracy: 0.9130 - val_loss: 0.1374 - val_accuracy: 0.9624
Epoch  2/10 loss: 0.1439 - accuracy: 0.9583 - val_loss: 0.1089 - val_accuracy: 0.9674
Epoch  3/10 loss: 0.1019 - accuracy: 0.9696 - val_loss: 0.0942 - val_accuracy: 0.9720
Epoch  4/10 loss: 0.0775 - accuracy: 0.9770 - val_loss: 0.0844 - val_accuracy: 0.9750
Epoch  5/10 loss: 0.0613 - accuracy: 0.9809 - val_loss: 0.0733 - val_accuracy: 0.9798
Epoch  6/10 loss: 0.0489 - accuracy: 0.9852 - val_loss: 0.0785 - val_accuracy: 0.9760
Epoch  7/10 loss: 0.0413 - accuracy: 0.9876 - val_loss: 0.0750 - val_accuracy: 0.9790
Epoch  8/10 loss: 0.0336 - accuracy: 0.9900 - val_loss: 0.0781 - val_accuracy: 0.9790
Epoch  9/10 loss: 0.0272 - accuracy: 0.9920 - val_loss: 0.0837 - val_accuracy: 0.9778
Epoch 10/10 loss: 0.0226 - accuracy: 0.9934 - val_loss: 0.0751 - val_accuracy: 0.9784
```
- `python3 mnist_cnn.py --cnn=F,H-100,D-0.5`
```
Epoch  1/10 loss: 0.4770 - accuracy: 0.8594 - val_loss: 0.1624 - val_accuracy: 0.9552
Epoch  2/10 loss: 0.2734 - accuracy: 0.9196 - val_loss: 0.1274 - val_accuracy: 0.9654
Epoch  3/10 loss: 0.2262 - accuracy: 0.9322 - val_loss: 0.1054 - val_accuracy: 0.9712
Epoch  4/10 loss: 0.2027 - accuracy: 0.9388 - val_loss: 0.0976 - val_accuracy: 0.9720
Epoch  5/10 loss: 0.1898 - accuracy: 0.9429 - val_loss: 0.0906 - val_accuracy: 0.9734
Epoch  6/10 loss: 0.1749 - accuracy: 0.9455 - val_loss: 0.0863 - val_accuracy: 0.9748
Epoch  7/10 loss: 0.1643 - accuracy: 0.9501 - val_loss: 0.0857 - val_accuracy: 0.9750
Epoch  8/10 loss: 0.1570 - accuracy: 0.9509 - val_loss: 0.0838 - val_accuracy: 0.9736
Epoch  9/10 loss: 0.1519 - accuracy: 0.9529 - val_loss: 0.0843 - val_accuracy: 0.9758
Epoch 10/10 loss: 0.1472 - accuracy: 0.9547 - val_loss: 0.0807 - val_accuracy: 0.9768
```
- `python3 mnist_cnn.py --cnn=F,H-200,D-0.5`
```
Epoch  1/10 loss: 0.3804 - accuracy: 0.8867 - val_loss: 0.1319 - val_accuracy: 0.9668
Epoch  2/10 loss: 0.1960 - accuracy: 0.9410 - val_loss: 0.1027 - val_accuracy: 0.9696
Epoch  3/10 loss: 0.1551 - accuracy: 0.9541 - val_loss: 0.0805 - val_accuracy: 0.9764
Epoch  4/10 loss: 0.1332 - accuracy: 0.9603 - val_loss: 0.0781 - val_accuracy: 0.9784
Epoch  5/10 loss: 0.1182 - accuracy: 0.9640 - val_loss: 0.0756 - val_accuracy: 0.9788
Epoch  6/10 loss: 0.1046 - accuracy: 0.9681 - val_loss: 0.0730 - val_accuracy: 0.9792
Epoch  7/10 loss: 0.1036 - accuracy: 0.9676 - val_loss: 0.0715 - val_accuracy: 0.9810
Epoch  8/10 loss: 0.0920 - accuracy: 0.9708 - val_loss: 0.0748 - val_accuracy: 0.9808
Epoch  9/10 loss: 0.0865 - accuracy: 0.9725 - val_loss: 0.0727 - val_accuracy: 0.9792
Epoch 10/10 loss: 0.0831 - accuracy: 0.9739 - val_loss: 0.0667 - val_accuracy: 0.9812
```
- `python3 mnist_cnn.py --cnn=C-8-3-1-same,C-8-3-1-same,M-3-2,C-16-3-1-same,C-16-3-1-same,M-3-2,F,H-100`
```
Epoch  1/10 loss: 0.1932 - accuracy: 0.9403 - val_loss: 0.0596 - val_accuracy: 0.9806
Epoch  2/10 loss: 0.0578 - accuracy: 0.9812 - val_loss: 0.0488 - val_accuracy: 0.9870
Epoch  3/10 loss: 0.0434 - accuracy: 0.9860 - val_loss: 0.0335 - val_accuracy: 0.9902
Epoch  4/10 loss: 0.0348 - accuracy: 0.9887 - val_loss: 0.0342 - val_accuracy: 0.9918
Epoch  5/10 loss: 0.0278 - accuracy: 0.9911 - val_loss: 0.0307 - val_accuracy: 0.9926
Epoch  6/10 loss: 0.0236 - accuracy: 0.9922 - val_loss: 0.0292 - val_accuracy: 0.9928
Epoch  7/10 loss: 0.0210 - accuracy: 0.9934 - val_loss: 0.0333 - val_accuracy: 0.9916
Epoch  8/10 loss: 0.0184 - accuracy: 0.9939 - val_loss: 0.0419 - val_accuracy: 0.9916
Epoch  9/10 loss: 0.0159 - accuracy: 0.9950 - val_loss: 0.0360 - val_accuracy: 0.9914
Epoch 10/10 loss: 0.0139 - accuracy: 0.9953 - val_loss: 0.0334 - val_accuracy: 0.9934
```
- `python3 mnist_cnn.py --cnn=CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100`
```
Epoch  1/10 loss: 0.1604 - accuracy: 0.9512 - val_loss: 0.0419 - val_accuracy: 0.9876
Epoch  2/10 loss: 0.0520 - accuracy: 0.9833 - val_loss: 0.0778 - val_accuracy: 0.9770
Epoch  3/10 loss: 0.0424 - accuracy: 0.9858 - val_loss: 0.0460 - val_accuracy: 0.9864
Epoch  4/10 loss: 0.0345 - accuracy: 0.9888 - val_loss: 0.0392 - val_accuracy: 0.9904
Epoch  5/10 loss: 0.0268 - accuracy: 0.9916 - val_loss: 0.0390 - val_accuracy: 0.9904
Epoch  6/10 loss: 0.0248 - accuracy: 0.9919 - val_loss: 0.0360 - val_accuracy: 0.9916
Epoch  7/10 loss: 0.0204 - accuracy: 0.9930 - val_loss: 0.0263 - val_accuracy: 0.9934
Epoch  8/10 loss: 0.0189 - accuracy: 0.9937 - val_loss: 0.0388 - val_accuracy: 0.9884
Epoch  9/10 loss: 0.0178 - accuracy: 0.9940 - val_loss: 0.0447 - val_accuracy: 0.9888
Epoch 10/10 loss: 0.0140 - accuracy: 0.9953 - val_loss: 0.0269 - val_accuracy: 0.9930
```
#### Examples End:
