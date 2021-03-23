### Assignment: mnist_multiple
#### Date: Deadline: Apr 05, 23:59
#### Points: 3 points
#### Examples: mnist_multiple_examples

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

#### Examples Start: mnist_multiple_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 mnist_multiple.py --batch_size=50`
```
Epoch 1/5 loss: 1.6499 - digit_1_loss: 0.6142 - digit_2_loss: 0.6227 - direct_prediction_loss: 0.4130 - direct_prediction_accuracy: 0.7896 - indirect_prediction_accuracy: 0.8972 - val_loss: 0.3579 - val_digit_1_loss: 0.1265 - val_digit_2_loss: 0.0724 - val_direct_prediction_loss: 0.1590 - val_direct_prediction_accuracy: 0.9428 - val_indirect_prediction_accuracy: 0.9800
Epoch 2/5 loss: 0.3472 - digit_1_loss: 0.0965 - digit_2_loss: 0.0988 - direct_prediction_loss: 0.1519 - direct_prediction_accuracy: 0.9452 - indirect_prediction_accuracy: 0.9788 - val_loss: 0.2222 - val_digit_1_loss: 0.0859 - val_digit_2_loss: 0.0555 - val_direct_prediction_loss: 0.0808 - val_direct_prediction_accuracy: 0.9724 - val_indirect_prediction_accuracy: 0.9872
Epoch 3/5 loss: 0.2184 - digit_1_loss: 0.0597 - digit_2_loss: 0.0624 - direct_prediction_loss: 0.0964 - direct_prediction_accuracy: 0.9643 - indirect_prediction_accuracy: 0.9868 - val_loss: 0.1976 - val_digit_1_loss: 0.0776 - val_digit_2_loss: 0.0610 - val_direct_prediction_loss: 0.0590 - val_direct_prediction_accuracy: 0.9824 - val_indirect_prediction_accuracy: 0.9856
Epoch 4/5 loss: 0.1540 - digit_1_loss: 0.0428 - digit_2_loss: 0.0454 - direct_prediction_loss: 0.0659 - direct_prediction_accuracy: 0.9781 - indirect_prediction_accuracy: 0.9889 - val_loss: 0.1753 - val_digit_1_loss: 0.0640 - val_digit_2_loss: 0.0523 - val_direct_prediction_loss: 0.0590 - val_direct_prediction_accuracy: 0.9776 - val_indirect_prediction_accuracy: 0.9876
Epoch 5/5 loss: 0.1253 - digit_1_loss: 0.0333 - digit_2_loss: 0.0337 - direct_prediction_loss: 0.0583 - direct_prediction_accuracy: 0.9806 - indirect_prediction_accuracy: 0.9914 - val_loss: 0.1596 - val_digit_1_loss: 0.0648 - val_digit_2_loss: 0.0525 - val_direct_prediction_loss: 0.0423 - val_direct_prediction_accuracy: 0.9880 - val_indirect_prediction_accuracy: 0.9908
loss: 0.1471 - digit_1_loss: 0.0429 - digit_2_loss: 0.0484 - direct_prediction_loss: 0.0558 - direct_prediction_accuracy: 0.9822 - indirect_prediction_accuracy: 0.9900
```
- `python3 mnist_multiple.py --batch_size=100`
```
Epoch 1/5 loss: 2.1134 - digit_1_loss: 0.8183 - digit_2_loss: 0.8250 - direct_prediction_loss: 0.4701 - direct_prediction_accuracy: 0.7570 - indirect_prediction_accuracy: 0.8735 - val_loss: 0.4835 - val_digit_1_loss: 0.1706 - val_digit_2_loss: 0.0993 - val_direct_prediction_loss: 0.2136 - val_direct_prediction_accuracy: 0.9168 - val_indirect_prediction_accuracy: 0.9700
Epoch 2/5 loss: 0.4881 - digit_1_loss: 0.1379 - digit_2_loss: 0.1396 - direct_prediction_loss: 0.2107 - direct_prediction_accuracy: 0.9159 - indirect_prediction_accuracy: 0.9706 - val_loss: 0.3022 - val_digit_1_loss: 0.1047 - val_digit_2_loss: 0.0659 - val_direct_prediction_loss: 0.1316 - val_direct_prediction_accuracy: 0.9500 - val_indirect_prediction_accuracy: 0.9832
Epoch 3/5 loss: 0.2938 - digit_1_loss: 0.0795 - digit_2_loss: 0.0825 - direct_prediction_loss: 0.1317 - direct_prediction_accuracy: 0.9493 - indirect_prediction_accuracy: 0.9825 - val_loss: 0.2150 - val_digit_1_loss: 0.0782 - val_digit_2_loss: 0.0586 - val_direct_prediction_loss: 0.0782 - val_direct_prediction_accuracy: 0.9688 - val_indirect_prediction_accuracy: 0.9888
Epoch 4/5 loss: 0.2026 - digit_1_loss: 0.0547 - digit_2_loss: 0.0607 - direct_prediction_loss: 0.0872 - direct_prediction_accuracy: 0.9693 - indirect_prediction_accuracy: 0.9881 - val_loss: 0.1970 - val_digit_1_loss: 0.0750 - val_digit_2_loss: 0.0543 - val_direct_prediction_loss: 0.0676 - val_direct_prediction_accuracy: 0.9748 - val_indirect_prediction_accuracy: 0.9868
Epoch 5/5 loss: 0.1618 - digit_1_loss: 0.0437 - digit_2_loss: 0.0470 - direct_prediction_loss: 0.0711 - direct_prediction_accuracy: 0.9753 - indirect_prediction_accuracy: 0.9893 - val_loss: 0.1735 - val_digit_1_loss: 0.0667 - val_digit_2_loss: 0.0507 - val_direct_prediction_loss: 0.0562 - val_direct_prediction_accuracy: 0.9816 - val_indirect_prediction_accuracy: 0.9896
loss: 0.1658 - digit_1_loss: 0.0469 - digit_2_loss: 0.0506 - direct_prediction_loss: 0.0683 - direct_prediction_accuracy: 0.9768 - indirect_prediction_accuracy: 0.9884
```
#### Examples End:
