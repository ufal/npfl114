### Assignment: mnist_multiple
#### Date: Deadline: Mar 21, 7:59
#### Points: 3 points
#### Tests: mnist_multiple_tests

In this assignment you will implement a model with multiple inputs and outputs.
Start with the [mnist_multiple.py](https://github.com/ufal/npfl114/tree/master/labs/04/mnist_multiple.py)
template and:
- The goal is to create a model, which given two input MNIST images predicts, if the
  digit on the first one is larger than on the second one.
- The model has four outputs:
  - direct prediction whether the first digit is larger than the second one,
  - digit classification for the first image,
  - digit classification for the second image,
  - indirect prediction comparing the digits predicted by the above two outputs.
- You need to implement:
  - the model, using multiple inputs, outputs, losses and metrics;
  - construction of two-image dataset examples using regular MNIST data via the `tf.data` API.

#### Tests Start: mnist_multiple_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_multiple.py --epochs=1 --batch_size=50`
```
loss: 0.9233 - digit_1_loss: 0.3067 - digit_2_loss: 0.3131 - direct_prediction_loss: 0.3036 - direct_prediction_accuracy: 0.8617 - indirect_prediction_accuracy: 0.9424 - val_loss: 0.3590 - val_digit_1_loss: 0.1264 - val_digit_2_loss: 0.0725 - val_direct_prediction_loss: 0.1601 - val_direct_prediction_accuracy: 0.9400 - val_indirect_prediction_accuracy: 0.9796
```
- `python3 mnist_multiple.py --epochs=1 --batch_size=100`
```
loss: 1.2151 - digit_1_loss: 0.4227 - digit_2_loss: 0.4280 - direct_prediction_loss: 0.3645 - direct_prediction_accuracy: 0.8301 - indirect_prediction_accuracy: 0.9257 - val_loss: 0.4846 - val_digit_1_loss: 0.1704 - val_digit_2_loss: 0.0990 - val_direct_prediction_loss: 0.2153 - val_direct_prediction_accuracy: 0.9164 - val_indirect_prediction_accuracy: 0.9700
```
#### Tests End:
