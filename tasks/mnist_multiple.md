### Assignment: mnist_multiple
#### Date: Deadline: Mar 20, 7:59 a.m.
#### Points: 3 points
#### Tests: mnist_multiple_tests

In this assignment you will implement a model with multiple inputs and outputs.
Start with the [mnist_multiple.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_multiple.py)
template and:

![mnist_multiple](//ufal.mff.cuni.cz/~straka/courses/npfl114/2223/tasks/figures/mnist_multiple.svgz)

- The goal is to create a model, which given two input MNIST images, compares if the
  digit on the first one is greater than on the second one.
- We perform this this comparison in two different ways:
  - first by directly predicting the comparison by the network (_direct comparison_),
  - then by first classifying the images into digits and then comparing these predictions (_indirect comparison_).
- The model has four outputs:
  - _direct comparison_ whether the first digit is greater than the second one,
  - digit classification for the first image,
  - digit classification for the second image,
  - _indirect comparison_ comparing the digits predicted by the above two outputs.
- You need to implement:
  - the model, using multiple inputs, outputs, losses and metrics;
  - construction of two-image dataset examples using regular MNIST data via the `tf.data` API.

#### Tests Start: mnist_multiple_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 mnist_multiple.py --epochs=1 --batch_size=50`
```
loss: 0.8763 - digit_1_loss: 0.2936 - digit_2_loss: 0.2972 - direct_comparison_loss: 0.2855 - direct_comparison_accuracy: 0.8711 - indirect_comparison_accuracy: 0.9450 - val_loss: 0.3029 - val_digit_1_loss: 0.1076 - val_digit_2_loss: 0.0644 - val_direct_comparison_loss: 0.1309 - val_direct_comparison_accuracy: 0.9556 - val_indirect_comparison_accuracy: 0.9828
```
2. `python3 mnist_multiple.py --epochs=1 --batch_size=100`
```
loss: 1.1698 - digit_1_loss: 0.4132 - digit_2_loss: 0.4140 - direct_comparison_loss: 0.3426 - direct_comparison_accuracy: 0.8390 - indirect_comparison_accuracy: 0.9270 - val_loss: 0.4259 - val_digit_1_loss: 0.1502 - val_digit_2_loss: 0.0884 - val_direct_comparison_loss: 0.1873 - val_direct_comparison_accuracy: 0.9296 - val_indirect_comparison_accuracy: 0.9744
```
#### Tests End:
