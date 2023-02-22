### Assignment: mnist_training
#### Date: Deadline: Mar 6, 7:59 a.m.
#### Points: 2 points
#### Examples: mnist_training_examples
#### Tests: mnist_training_tests

This exercise should teach you using different optimizers, learning rates,
and learning rate decays. Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl114/tree/master/labs/02/mnist_training.py)
template and implement the following:
- Using specified optimizer (either `SGD` or `Adam`).
- Optionally using momentum for the `SGD` optimizer.
- Using specified learning rate for the optimizer.
- Optionally use a given learning rate schedule. The schedule can be either
  `linear`, `exponential`, or `cosine`. If a schedule is specified, you also
  get a final learning rate, and the learning rate should be gradually decresed
  during training to reach the final learning rate just after the training
  (i.e., the first update after the training would use exactly the final learning rate).

#### Examples Start: mnist_training_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01`
```
Epoch  1/10 loss: 0.8214 - accuracy: 0.7996 - val_loss: 0.3673 - val_accuracy: 0.9096
Epoch  2/10 loss: 0.3950 - accuracy: 0.8930 - val_loss: 0.2821 - val_accuracy: 0.9250
Epoch  3/10 loss: 0.3346 - accuracy: 0.9064 - val_loss: 0.2472 - val_accuracy: 0.9326
Epoch  4/10 loss: 0.3018 - accuracy: 0.9150 - val_loss: 0.2268 - val_accuracy: 0.9394
Epoch  5/10 loss: 0.2786 - accuracy: 0.9215 - val_loss: 0.2113 - val_accuracy: 0.9416
Epoch  6/10 loss: 0.2603 - accuracy: 0.9271 - val_loss: 0.1996 - val_accuracy: 0.9454
Epoch  7/10 loss: 0.2448 - accuracy: 0.9313 - val_loss: 0.1879 - val_accuracy: 0.9500
Epoch  8/10 loss: 0.2314 - accuracy: 0.9352 - val_loss: 0.1819 - val_accuracy: 0.9512
Epoch  9/10 loss: 0.2199 - accuracy: 0.9384 - val_loss: 0.1720 - val_accuracy: 0.9558
Epoch 10/10 loss: 0.2091 - accuracy: 0.9416 - val_loss: 0.1655 - val_accuracy: 0.9582
```
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch  1/10 loss: 0.3626 - accuracy: 0.8976 - val_loss: 0.1689 - val_accuracy: 0.9556
Epoch  2/10 loss: 0.1813 - accuracy: 0.9486 - val_loss: 0.1278 - val_accuracy: 0.9668
Epoch  3/10 loss: 0.1324 - accuracy: 0.9622 - val_loss: 0.1060 - val_accuracy: 0.9706
Epoch  4/10 loss: 0.1053 - accuracy: 0.9704 - val_loss: 0.0916 - val_accuracy: 0.9742
Epoch  5/10 loss: 0.0866 - accuracy: 0.9753 - val_loss: 0.0862 - val_accuracy: 0.9766
Epoch  6/10 loss: 0.0732 - accuracy: 0.9793 - val_loss: 0.0806 - val_accuracy: 0.9774
Epoch  7/10 loss: 0.0637 - accuracy: 0.9825 - val_loss: 0.0756 - val_accuracy: 0.9806
Epoch  8/10 loss: 0.0547 - accuracy: 0.9851 - val_loss: 0.0740 - val_accuracy: 0.9794
Epoch  9/10 loss: 0.0486 - accuracy: 0.9867 - val_loss: 0.0781 - val_accuracy: 0.9768
Epoch 10/10 loss: 0.0430 - accuracy: 0.9886 - val_loss: 0.0731 - val_accuracy: 0.9790
```
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.1`
```
Epoch  1/10 loss: 0.3515 - accuracy: 0.9008 - val_loss: 0.1660 - val_accuracy: 0.9564
Epoch  2/10 loss: 0.1788 - accuracy: 0.9488 - val_loss: 0.1267 - val_accuracy: 0.9668
Epoch  3/10 loss: 0.1307 - accuracy: 0.9624 - val_loss: 0.1006 - val_accuracy: 0.9734
Epoch  4/10 loss: 0.1039 - accuracy: 0.9711 - val_loss: 0.0902 - val_accuracy: 0.9736
Epoch  5/10 loss: 0.0856 - accuracy: 0.9755 - val_loss: 0.0845 - val_accuracy: 0.9776
Epoch  6/10 loss: 0.0729 - accuracy: 0.9789 - val_loss: 0.0841 - val_accuracy: 0.9762
Epoch  7/10 loss: 0.0628 - accuracy: 0.9827 - val_loss: 0.0742 - val_accuracy: 0.9812
Epoch  8/10 loss: 0.0546 - accuracy: 0.9850 - val_loss: 0.0746 - val_accuracy: 0.9790
Epoch  9/10 loss: 0.0488 - accuracy: 0.9866 - val_loss: 0.0756 - val_accuracy: 0.9780
Epoch 10/10 loss: 0.0426 - accuracy: 0.9887 - val_loss: 0.0711 - val_accuracy: 0.9784
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.001`
```
Epoch  1/10 loss: 0.2732 - accuracy: 0.9221 - val_loss: 0.1186 - val_accuracy: 0.9674
Epoch  2/10 loss: 0.1156 - accuracy: 0.9662 - val_loss: 0.0921 - val_accuracy: 0.9716
Epoch  3/10 loss: 0.0776 - accuracy: 0.9772 - val_loss: 0.0785 - val_accuracy: 0.9764
Epoch  4/10 loss: 0.0569 - accuracy: 0.9831 - val_loss: 0.0795 - val_accuracy: 0.9756
Epoch  5/10 loss: 0.0428 - accuracy: 0.9866 - val_loss: 0.0736 - val_accuracy: 0.9788
Epoch  6/10 loss: 0.0324 - accuracy: 0.9900 - val_loss: 0.0749 - val_accuracy: 0.9806
Epoch  7/10 loss: 0.0265 - accuracy: 0.9921 - val_loss: 0.0781 - val_accuracy: 0.9782
Epoch  8/10 loss: 0.0203 - accuracy: 0.9942 - val_loss: 0.0886 - val_accuracy: 0.9776
Epoch  9/10 loss: 0.0157 - accuracy: 0.9953 - val_loss: 0.0830 - val_accuracy: 0.9786
Epoch 10/10 loss: 0.0136 - accuracy: 0.9958 - val_loss: 0.0878 - val_accuracy: 0.9778
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01`
```
Epoch  1/10 loss: 0.2312 - accuracy: 0.9309 - val_loss: 0.1286 - val_accuracy: 0.9648
Epoch  2/10 loss: 0.1384 - accuracy: 0.9607 - val_loss: 0.1295 - val_accuracy: 0.9638
Epoch  3/10 loss: 0.1251 - accuracy: 0.9655 - val_loss: 0.1784 - val_accuracy: 0.9594
Epoch  4/10 loss: 0.1079 - accuracy: 0.9701 - val_loss: 0.1693 - val_accuracy: 0.9608
Epoch  5/10 loss: 0.0988 - accuracy: 0.9729 - val_loss: 0.1524 - val_accuracy: 0.9676
Epoch  6/10 loss: 0.0950 - accuracy: 0.9747 - val_loss: 0.1813 - val_accuracy: 0.9698
Epoch  7/10 loss: 0.0938 - accuracy: 0.9764 - val_loss: 0.1850 - val_accuracy: 0.9654
Epoch  8/10 loss: 0.0849 - accuracy: 0.9781 - val_loss: 0.1919 - val_accuracy: 0.9670
Epoch  9/10 loss: 0.0833 - accuracy: 0.9792 - val_loss: 0.1739 - val_accuracy: 0.9712
Epoch 10/10 loss: 0.0744 - accuracy: 0.9815 - val_loss: 0.1840 - val_accuracy: 0.9690
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch  1/10 loss: 0.2204 - accuracy: 0.9335 - val_loss: 0.1276 - val_accuracy: 0.9656
Epoch  2/10 loss: 0.1126 - accuracy: 0.9672 - val_loss: 0.1180 - val_accuracy: 0.9660
Epoch  3/10 loss: 0.0745 - accuracy: 0.9767 - val_loss: 0.0989 - val_accuracy: 0.9750
Epoch  4/10 loss: 0.0495 - accuracy: 0.9843 - val_loss: 0.0898 - val_accuracy: 0.9780
Epoch  5/10 loss: 0.0326 - accuracy: 0.9899 - val_loss: 0.0970 - val_accuracy: 0.9788
Epoch  6/10 loss: 0.0197 - accuracy: 0.9936 - val_loss: 0.1005 - val_accuracy: 0.9808
Epoch  7/10 loss: 0.0133 - accuracy: 0.9955 - val_loss: 0.0857 - val_accuracy: 0.9812
Epoch  8/10 loss: 0.0067 - accuracy: 0.9982 - val_loss: 0.0976 - val_accuracy: 0.9804
Epoch  9/10 loss: 0.0042 - accuracy: 0.9991 - val_loss: 0.1056 - val_accuracy: 0.9804
Epoch 10/10 loss: 0.0023 - accuracy: 0.9997 - val_loss: 0.0931 - val_accuracy: 0.9822
Next learning rate to be used: 0.001
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch  1/10 loss: 0.2299 - accuracy: 0.9312 - val_loss: 0.1309 - val_accuracy: 0.9620
Epoch  2/10 loss: 0.1266 - accuracy: 0.9632 - val_loss: 0.1174 - val_accuracy: 0.9702
Epoch  3/10 loss: 0.0958 - accuracy: 0.9724 - val_loss: 0.1129 - val_accuracy: 0.9730
Epoch  4/10 loss: 0.0730 - accuracy: 0.9775 - val_loss: 0.1223 - val_accuracy: 0.9700
Epoch  5/10 loss: 0.0504 - accuracy: 0.9847 - val_loss: 0.1046 - val_accuracy: 0.9758
Epoch  6/10 loss: 0.0338 - accuracy: 0.9895 - val_loss: 0.1225 - val_accuracy: 0.9766
Epoch  7/10 loss: 0.0239 - accuracy: 0.9925 - val_loss: 0.1043 - val_accuracy: 0.9784
Epoch  8/10 loss: 0.0108 - accuracy: 0.9964 - val_loss: 0.1035 - val_accuracy: 0.9808
Epoch  9/10 loss: 0.0050 - accuracy: 0.9985 - val_loss: 0.0912 - val_accuracy: 0.9822
Epoch 10/10 loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.0920 - val_accuracy: 0.9828
Next learning rate to be used: 0.0001
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch  1/10 loss: 0.2307 - accuracy: 0.9302 - val_loss: 0.1340 - val_accuracy: 0.9620
Epoch  2/10 loss: 0.1377 - accuracy: 0.9608 - val_loss: 0.1398 - val_accuracy: 0.9640
Epoch  3/10 loss: 0.1089 - accuracy: 0.9676 - val_loss: 0.1089 - val_accuracy: 0.9738
Epoch  4/10 loss: 0.0774 - accuracy: 0.9775 - val_loss: 0.1198 - val_accuracy: 0.9710
Epoch  5/10 loss: 0.0517 - accuracy: 0.9844 - val_loss: 0.1100 - val_accuracy: 0.9758
Epoch  6/10 loss: 0.0333 - accuracy: 0.9890 - val_loss: 0.1036 - val_accuracy: 0.9786
Epoch  7/10 loss: 0.0181 - accuracy: 0.9941 - val_loss: 0.0949 - val_accuracy: 0.9814
Epoch  8/10 loss: 0.0091 - accuracy: 0.9973 - val_loss: 0.0930 - val_accuracy: 0.9812
Epoch  9/10 loss: 0.0050 - accuracy: 0.9987 - val_loss: 0.0971 - val_accuracy: 0.9826
Epoch 10/10 loss: 0.0036 - accuracy: 0.9992 - val_loss: 0.0965 - val_accuracy: 0.9824
Next learning rate to be used: 0.0001
```
#### Examples End:
#### Tests Start: mnist_training_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01`
```
loss: 0.8214 - accuracy: 0.7996 - val_loss: 0.3673 - val_accuracy: 0.9096
```
- `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
loss: 0.3626 - accuracy: 0.8976 - val_loss: 0.1689 - val_accuracy: 0.9556
```
- `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.1`
```
loss: 0.3515 - accuracy: 0.9008 - val_loss: 0.1660 - val_accuracy: 0.9564
```
- `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.001`
```
loss: 0.2732 - accuracy: 0.9221 - val_loss: 0.1186 - val_accuracy: 0.9674
```
- `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.01`
```
loss: 0.2312 - accuracy: 0.9309 - val_loss: 0.1286 - val_accuracy: 0.9648
```
- `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch 1/2 loss: 0.1962 - accuracy: 0.9398 - val_loss: 0.1026 - val_accuracy: 0.9728
Epoch 2/2 loss: 0.0672 - accuracy: 0.9788 - val_loss: 0.0735 - val_accuracy: 0.9788
Next learning rate to be used: 0.001
```
- `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch 1/2 loss: 0.2106 - accuracy: 0.9369 - val_loss: 0.1174 - val_accuracy: 0.9664
Epoch 2/2 loss: 0.0715 - accuracy: 0.9775 - val_loss: 0.0745 - val_accuracy: 0.9778
Next learning rate to be used: 0.0001
```
- `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch 1/2 loss: 0.2158 - accuracy: 0.9346 - val_loss: 0.1231 - val_accuracy: 0.9670
Epoch 2/2 loss: 0.0694 - accuracy: 0.9781 - val_loss: 0.0746 - val_accuracy: 0.9786
Next learning rate to be used: 0.0001
```
#### Tests End:
