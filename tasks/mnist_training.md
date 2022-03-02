### Assignment: mnist_training
#### Date: Deadline: Mar 14, 7:59 a.m.
#### Points: 2 points
#### Examples: mnist_training_examples
#### Tests: mnist_training_tests

This exercise should teach you using different optimizers, learning rates,
and learning rate decays. Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl114/tree/master/labs/03/mnist_training.py)
template and implement the following:
- Using specified optimizer (either `SGD` or `Adam`).
- Optionally using momentum for the `SGD` optimizer.
- Using specified learning rate for the optimizer.
- Optionally use a given learning rate schedule. The schedule can be either
  `exponential` or `linear` (with degree 1, so linear time decay).
  Additionally, the final learning rate is given and the decay should gradually
  decrease the learning rate to reach the final learning rate just after the
  training.

#### Examples Start: mnist_training_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01`
```
Epoch  1/10 loss: 0.7989 - accuracy: 0.8098 - val_loss: 0.3662 - val_accuracy: 0.9146
Epoch  2/10 loss: 0.3991 - accuracy: 0.8919 - val_loss: 0.2848 - val_accuracy: 0.9258
Epoch  3/10 loss: 0.3382 - accuracy: 0.9054 - val_loss: 0.2496 - val_accuracy: 0.9350
Epoch  4/10 loss: 0.3049 - accuracy: 0.9144 - val_loss: 0.2292 - val_accuracy: 0.9390
Epoch  5/10 loss: 0.2811 - accuracy: 0.9216 - val_loss: 0.2131 - val_accuracy: 0.9426
Epoch  6/10 loss: 0.2623 - accuracy: 0.9268 - val_loss: 0.2003 - val_accuracy: 0.9464
Epoch  7/10 loss: 0.2461 - accuracy: 0.9315 - val_loss: 0.1882 - val_accuracy: 0.9500
Epoch  8/10 loss: 0.2323 - accuracy: 0.9353 - val_loss: 0.1821 - val_accuracy: 0.9522
Epoch  9/10 loss: 0.2204 - accuracy: 0.9386 - val_loss: 0.1715 - val_accuracy: 0.9560
Epoch 10/10 loss: 0.2094 - accuracy: 0.9413 - val_loss: 0.1650 - val_accuracy: 0.9572 - val_test_loss: 0.1978 - val_test_accuracy: 0.9441
```
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch  1/10 loss: 0.3617 - accuracy: 0.8973 - val_loss: 0.1684 - val_accuracy: 0.9560
Epoch  2/10 loss: 0.1803 - accuracy: 0.9490 - val_loss: 0.1274 - val_accuracy: 0.9644
Epoch  3/10 loss: 0.1319 - accuracy: 0.9625 - val_loss: 0.1051 - val_accuracy: 0.9706
Epoch  4/10 loss: 0.1048 - accuracy: 0.9709 - val_loss: 0.0922 - val_accuracy: 0.9746
Epoch  5/10 loss: 0.0864 - accuracy: 0.9756 - val_loss: 0.0844 - val_accuracy: 0.9782
Epoch  6/10 loss: 0.0731 - accuracy: 0.9794 - val_loss: 0.0791 - val_accuracy: 0.9784
Epoch  7/10 loss: 0.0633 - accuracy: 0.9825 - val_loss: 0.0738 - val_accuracy: 0.9818
Epoch  8/10 loss: 0.0550 - accuracy: 0.9848 - val_loss: 0.0746 - val_accuracy: 0.9796
Epoch  9/10 loss: 0.0485 - accuracy: 0.9866 - val_loss: 0.0758 - val_accuracy: 0.9796
Epoch 10/10 loss: 0.0429 - accuracy: 0.9888 - val_loss: 0.0704 - val_accuracy: 0.9806 - val_test_loss: 0.0677 - val_test_accuracy: 0.9789
```
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.1`
```
Epoch  1/10 loss: 0.3502 - accuracy: 0.9021 - val_loss: 0.1679 - val_accuracy: 0.9576
Epoch  2/10 loss: 0.1784 - accuracy: 0.9492 - val_loss: 0.1265 - val_accuracy: 0.9646
Epoch  3/10 loss: 0.1303 - accuracy: 0.9629 - val_loss: 0.0994 - val_accuracy: 0.9724
Epoch  4/10 loss: 0.1033 - accuracy: 0.9714 - val_loss: 0.0891 - val_accuracy: 0.9754
Epoch  5/10 loss: 0.0848 - accuracy: 0.9757 - val_loss: 0.0847 - val_accuracy: 0.9776
Epoch  6/10 loss: 0.0721 - accuracy: 0.9794 - val_loss: 0.0802 - val_accuracy: 0.9778
Epoch  7/10 loss: 0.0620 - accuracy: 0.9829 - val_loss: 0.0724 - val_accuracy: 0.9818
Epoch  8/10 loss: 0.0541 - accuracy: 0.9853 - val_loss: 0.0724 - val_accuracy: 0.9808
Epoch  9/10 loss: 0.0480 - accuracy: 0.9868 - val_loss: 0.0745 - val_accuracy: 0.9796
Epoch 10/10 loss: 0.0421 - accuracy: 0.9890 - val_loss: 0.0665 - val_accuracy: 0.9824 - val_test_loss: 0.0658 - val_test_accuracy: 0.9800
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.001`
```
Epoch  1/10 loss: 0.2699 - accuracy: 0.9231 - val_loss: 0.1166 - val_accuracy: 0.9686
Epoch  2/10 loss: 0.1139 - accuracy: 0.9665 - val_loss: 0.0921 - val_accuracy: 0.9748
Epoch  3/10 loss: 0.0769 - accuracy: 0.9773 - val_loss: 0.0831 - val_accuracy: 0.9774
Epoch  4/10 loss: 0.0561 - accuracy: 0.9833 - val_loss: 0.0758 - val_accuracy: 0.9780
Epoch  5/10 loss: 0.0425 - accuracy: 0.9872 - val_loss: 0.0732 - val_accuracy: 0.9800
Epoch  6/10 loss: 0.0312 - accuracy: 0.9910 - val_loss: 0.0838 - val_accuracy: 0.9804
Epoch  7/10 loss: 0.0268 - accuracy: 0.9918 - val_loss: 0.0776 - val_accuracy: 0.9812
Epoch  8/10 loss: 0.0194 - accuracy: 0.9941 - val_loss: 0.0739 - val_accuracy: 0.9818
Epoch  9/10 loss: 0.0154 - accuracy: 0.9957 - val_loss: 0.0796 - val_accuracy: 0.9816
Epoch 10/10 loss: 0.0128 - accuracy: 0.9962 - val_loss: 0.0828 - val_accuracy: 0.9778 - val_test_loss: 0.0762 - val_test_accuracy: 0.9786
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01`
```
Epoch  1/10 loss: 0.2354 - accuracy: 0.9290 - val_loss: 0.1425 - val_accuracy: 0.9576
Epoch  2/10 loss: 0.1450 - accuracy: 0.9590 - val_loss: 0.1551 - val_accuracy: 0.9584
Epoch  3/10 loss: 0.1240 - accuracy: 0.9647 - val_loss: 0.1432 - val_accuracy: 0.9682
Epoch  4/10 loss: 0.1161 - accuracy: 0.9697 - val_loss: 0.1400 - val_accuracy: 0.9626
Epoch  5/10 loss: 0.1081 - accuracy: 0.9718 - val_loss: 0.1329 - val_accuracy: 0.9688
Epoch  6/10 loss: 0.0908 - accuracy: 0.9771 - val_loss: 0.1663 - val_accuracy: 0.9688
Epoch  7/10 loss: 0.0936 - accuracy: 0.9767 - val_loss: 0.1644 - val_accuracy: 0.9670
Epoch  8/10 loss: 0.0872 - accuracy: 0.9784 - val_loss: 0.1550 - val_accuracy: 0.9686
Epoch  9/10 loss: 0.0817 - accuracy: 0.9798 - val_loss: 0.2147 - val_accuracy: 0.9642
Epoch 10/10 loss: 0.0779 - accuracy: 0.9807 - val_loss: 0.1981 - val_accuracy: 0.9718 - val_test_loss: 0.1910 - val_test_accuracy: 0.9726
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch  1/10 loss: 0.2235 - accuracy: 0.9319 - val_loss: 0.1255 - val_accuracy: 0.9652
Epoch  2/10 loss: 0.1145 - accuracy: 0.9659 - val_loss: 0.1273 - val_accuracy: 0.9666
Epoch  3/10 loss: 0.0761 - accuracy: 0.9762 - val_loss: 0.0905 - val_accuracy: 0.9778
Epoch  4/10 loss: 0.0514 - accuracy: 0.9842 - val_loss: 0.1031 - val_accuracy: 0.9736
Epoch  5/10 loss: 0.0323 - accuracy: 0.9893 - val_loss: 0.1046 - val_accuracy: 0.9772
Epoch  6/10 loss: 0.0189 - accuracy: 0.9938 - val_loss: 0.1010 - val_accuracy: 0.9794
Epoch  7/10 loss: 0.0127 - accuracy: 0.9959 - val_loss: 0.1019 - val_accuracy: 0.9790
Epoch  8/10 loss: 0.0073 - accuracy: 0.9977 - val_loss: 0.1066 - val_accuracy: 0.9792
Epoch  9/10 loss: 0.0039 - accuracy: 0.9990 - val_loss: 0.1049 - val_accuracy: 0.9806
Epoch 10/10 loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.1057 - val_accuracy: 0.9798 - val_test_loss: 0.0868 - val_test_accuracy: 0.9809
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch  1/10 loss: 0.2292 - accuracy: 0.9303 - val_loss: 0.1176 - val_accuracy: 0.9634
Epoch  2/10 loss: 0.1291 - accuracy: 0.9621 - val_loss: 0.1193 - val_accuracy: 0.9658
Epoch  3/10 loss: 0.0973 - accuracy: 0.9719 - val_loss: 0.1094 - val_accuracy: 0.9712
Epoch  4/10 loss: 0.0694 - accuracy: 0.9796 - val_loss: 0.1408 - val_accuracy: 0.9656
Epoch  5/10 loss: 0.0523 - accuracy: 0.9840 - val_loss: 0.1234 - val_accuracy: 0.9704
Epoch  6/10 loss: 0.0346 - accuracy: 0.9889 - val_loss: 0.1381 - val_accuracy: 0.9740
Epoch  7/10 loss: 0.0249 - accuracy: 0.9922 - val_loss: 0.1105 - val_accuracy: 0.9776
Epoch  8/10 loss: 0.0105 - accuracy: 0.9968 - val_loss: 0.1115 - val_accuracy: 0.9780
Epoch  9/10 loss: 0.0050 - accuracy: 0.9985 - val_loss: 0.1144 - val_accuracy: 0.9800
Epoch 10/10 loss: 0.0023 - accuracy: 0.9995 - val_loss: 0.1127 - val_accuracy: 0.9788 - val_test_loss: 0.0975 - val_test_accuracy: 0.9812
```
#### Examples End:
#### Tests Start: mnist_training_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01`
```
loss: 0.7989 - accuracy: 0.8098 - val_loss: 0.3662 - val_accuracy: 0.9146 - val_test_loss: 0.4247 - val_test_accuracy: 0.8926
```
- `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
loss: 0.3617 - accuracy: 0.8973 - val_loss: 0.1684 - val_accuracy: 0.9560 - val_test_loss: 0.2011 - val_test_accuracy: 0.9456
```
- `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.1`
```
loss: 0.3502 - accuracy: 0.9021 - val_loss: 0.1679 - val_accuracy: 0.9576 - val_test_loss: 0.1996 - val_test_accuracy: 0.9454
```
- `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.001`
```
loss: 0.2699 - accuracy: 0.9231 - val_loss: 0.1166 - val_accuracy: 0.9686 - val_test_loss: 0.1385 - val_test_accuracy: 0.9605
```
- `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.01`
```
loss: 0.2354 - accuracy: 0.9290 - val_loss: 0.1425 - val_accuracy: 0.9576 - val_test_loss: 0.1692 - val_test_accuracy: 0.9469
```
- `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch 1/2 loss: 0.1961 - accuracy: 0.9400 - val_loss: 0.0890 - val_accuracy: 0.9728
Epoch 2/2 loss: 0.0663 - accuracy: 0.9792 - val_loss: 0.0675 - val_accuracy: 0.9790 - val_test_loss: 0.0721 - val_test_accuracy: 0.9773
Final learning rate: 0.001
```
- `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch 1/2 loss: 0.2111 - accuracy: 0.9356 - val_loss: 0.1017 - val_accuracy: 0.9690
Epoch 2/2 loss: 0.0701 - accuracy: 0.9781 - val_loss: 0.0708 - val_accuracy: 0.9790 - val_test_loss: 0.0693 - val_test_accuracy: 0.9779
Final learning rate: 0.0001
```
#### Tests End:
