### Assignment: mnist_training
#### Date: Deadline: Mar 14, 7:59 a.m.
#### Points: 3 points
#### Examples: mnist_training_examples

This exercise should teach you using different optimizers, learning rates,
and learning rate decays. Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl114/tree/master/labs/02/mnist_training.py)
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
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01`
```
Epoch  1/10 loss: 1.2077 - accuracy: 0.6998 - val_loss: 0.3662 - val_accuracy: 0.9146
Epoch  2/10 loss: 0.4205 - accuracy: 0.8871 - val_loss: 0.2848 - val_accuracy: 0.9258
Epoch  3/10 loss: 0.3458 - accuracy: 0.9038 - val_loss: 0.2496 - val_accuracy: 0.9350
Epoch  4/10 loss: 0.3115 - accuracy: 0.9139 - val_loss: 0.2292 - val_accuracy: 0.9390
Epoch  5/10 loss: 0.2862 - accuracy: 0.9202 - val_loss: 0.2131 - val_accuracy: 0.9426
Epoch  6/10 loss: 0.2698 - accuracy: 0.9231 - val_loss: 0.2003 - val_accuracy: 0.9464
Epoch  7/10 loss: 0.2489 - accuracy: 0.9296 - val_loss: 0.1881 - val_accuracy: 0.9500
Epoch  8/10 loss: 0.2344 - accuracy: 0.9331 - val_loss: 0.1821 - val_accuracy: 0.9522
Epoch  9/10 loss: 0.2203 - accuracy: 0.9385 - val_loss: 0.1715 - val_accuracy: 0.9560
Epoch 10/10 loss: 0.2130 - accuracy: 0.9397 - val_loss: 0.1650 - val_accuracy: 0.9572
loss: 0.1977 - accuracy: 0.9442
```
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch  1/10 loss: 0.5876 - accuracy: 0.8309 - val_loss: 0.1684 - val_accuracy: 0.9560
Epoch  2/10 loss: 0.1929 - accuracy: 0.9458 - val_loss: 0.1274 - val_accuracy: 0.9644
Epoch  3/10 loss: 0.1370 - accuracy: 0.9617 - val_loss: 0.1051 - val_accuracy: 0.9706
Epoch  4/10 loss: 0.1073 - accuracy: 0.9696 - val_loss: 0.0922 - val_accuracy: 0.9746
Epoch  5/10 loss: 0.0870 - accuracy: 0.9754 - val_loss: 0.0844 - val_accuracy: 0.9782
Epoch  6/10 loss: 0.0740 - accuracy: 0.9798 - val_loss: 0.0790 - val_accuracy: 0.9782
Epoch  7/10 loss: 0.0616 - accuracy: 0.9827 - val_loss: 0.0738 - val_accuracy: 0.9820
Epoch  8/10 loss: 0.0546 - accuracy: 0.9853 - val_loss: 0.0749 - val_accuracy: 0.9796
Epoch  9/10 loss: 0.0450 - accuracy: 0.9878 - val_loss: 0.0762 - val_accuracy: 0.9798
Epoch 10/10 loss: 0.0438 - accuracy: 0.9885 - val_loss: 0.0703 - val_accuracy: 0.9806
loss: 0.0675 - accuracy: 0.9794
```
- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.1`
```
Epoch  1/10 loss: 0.5462 - accuracy: 0.8503 - val_loss: 0.1677 - val_accuracy: 0.9572
Epoch  2/10 loss: 0.1909 - accuracy: 0.9459 - val_loss: 0.1267 - val_accuracy: 0.9648
Epoch  3/10 loss: 0.1361 - accuracy: 0.9615 - val_loss: 0.0994 - val_accuracy: 0.9724
Epoch  4/10 loss: 0.1057 - accuracy: 0.9699 - val_loss: 0.0890 - val_accuracy: 0.9762
Epoch  5/10 loss: 0.0851 - accuracy: 0.9762 - val_loss: 0.0844 - val_accuracy: 0.9784
Epoch  6/10 loss: 0.0730 - accuracy: 0.9796 - val_loss: 0.0800 - val_accuracy: 0.9784
Epoch  7/10 loss: 0.0604 - accuracy: 0.9833 - val_loss: 0.0725 - val_accuracy: 0.9814
Epoch  8/10 loss: 0.0536 - accuracy: 0.9859 - val_loss: 0.0726 - val_accuracy: 0.9796
Epoch  9/10 loss: 0.0444 - accuracy: 0.9886 - val_loss: 0.0744 - val_accuracy: 0.9802
Epoch 10/10 loss: 0.0430 - accuracy: 0.9883 - val_loss: 0.0665 - val_accuracy: 0.9822
loss: 0.0658 - accuracy: 0.9800
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.001`
```
Epoch  1/10 loss: 0.4529 - accuracy: 0.8712 - val_loss: 0.1166 - val_accuracy: 0.9686
Epoch  2/10 loss: 0.1205 - accuracy: 0.9648 - val_loss: 0.0921 - val_accuracy: 0.9748
Epoch  3/10 loss: 0.0763 - accuracy: 0.9775 - val_loss: 0.0831 - val_accuracy: 0.9774
Epoch  4/10 loss: 0.0540 - accuracy: 0.9844 - val_loss: 0.0758 - val_accuracy: 0.9780
Epoch  5/10 loss: 0.0408 - accuracy: 0.9879 - val_loss: 0.0733 - val_accuracy: 0.9808
Epoch  6/10 loss: 0.0298 - accuracy: 0.9919 - val_loss: 0.0833 - val_accuracy: 0.9810
Epoch  7/10 loss: 0.0238 - accuracy: 0.9936 - val_loss: 0.0761 - val_accuracy: 0.9814
Epoch  8/10 loss: 0.0169 - accuracy: 0.9950 - val_loss: 0.0760 - val_accuracy: 0.9796
Epoch  9/10 loss: 0.0132 - accuracy: 0.9966 - val_loss: 0.0810 - val_accuracy: 0.9814
Epoch 10/10 loss: 0.0116 - accuracy: 0.9968 - val_loss: 0.0913 - val_accuracy: 0.9782
loss: 0.0812 - accuracy: 0.9784
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01`
```
Epoch  1/10 loss: 0.3453 - accuracy: 0.8944 - val_loss: 0.1442 - val_accuracy: 0.9586
Epoch  2/10 loss: 0.1415 - accuracy: 0.9585 - val_loss: 0.1317 - val_accuracy: 0.9638
Epoch  3/10 loss: 0.1126 - accuracy: 0.9685 - val_loss: 0.1323 - val_accuracy: 0.9646
Epoch  4/10 loss: 0.0977 - accuracy: 0.9720 - val_loss: 0.1397 - val_accuracy: 0.9684
Epoch  5/10 loss: 0.0938 - accuracy: 0.9744 - val_loss: 0.1374 - val_accuracy: 0.9708
Epoch  6/10 loss: 0.0864 - accuracy: 0.9755 - val_loss: 0.2143 - val_accuracy: 0.9618
Epoch  7/10 loss: 0.0863 - accuracy: 0.9773 - val_loss: 0.1833 - val_accuracy: 0.9696
Epoch  8/10 loss: 0.0741 - accuracy: 0.9801 - val_loss: 0.1747 - val_accuracy: 0.9716
Epoch  9/10 loss: 0.0734 - accuracy: 0.9815 - val_loss: 0.2182 - val_accuracy: 0.9668
Epoch 10/10 loss: 0.0715 - accuracy: 0.9828 - val_loss: 0.2157 - val_accuracy: 0.9698
loss: 0.2383 - accuracy: 0.9687
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch  1/10 loss: 0.3396 - accuracy: 0.8952 - val_loss: 0.1255 - val_accuracy: 0.9652
Epoch  2/10 loss: 0.1132 - accuracy: 0.9654 - val_loss: 0.1273 - val_accuracy: 0.9666
Epoch  3/10 loss: 0.0714 - accuracy: 0.9776 - val_loss: 0.0896 - val_accuracy: 0.9768
Epoch  4/10 loss: 0.0467 - accuracy: 0.9854 - val_loss: 0.0970 - val_accuracy: 0.9756
Epoch  5/10 loss: 0.0315 - accuracy: 0.9896 - val_loss: 0.1041 - val_accuracy: 0.9788
Epoch  6/10 loss: 0.0193 - accuracy: 0.9934 - val_loss: 0.1029 - val_accuracy: 0.9790
Epoch  7/10 loss: 0.0121 - accuracy: 0.9961 - val_loss: 0.0926 - val_accuracy: 0.9802
Epoch  8/10 loss: 0.0061 - accuracy: 0.9983 - val_loss: 0.1044 - val_accuracy: 0.9802
Epoch  9/10 loss: 0.0035 - accuracy: 0.9992 - val_loss: 0.0992 - val_accuracy: 0.9806
Epoch 10/10 loss: 0.0029 - accuracy: 0.9994 - val_loss: 0.1052 - val_accuracy: 0.9816
loss: 0.0880 - accuracy: 0.9797
Final learning rate: 0.001
```
- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch  1/10 loss: 0.3428 - accuracy: 0.8944 - val_loss: 0.1176 - val_accuracy: 0.9634
Epoch  2/10 loss: 0.1229 - accuracy: 0.9632 - val_loss: 0.1303 - val_accuracy: 0.9642
Epoch  3/10 loss: 0.0920 - accuracy: 0.9728 - val_loss: 0.1064 - val_accuracy: 0.9724
Epoch  4/10 loss: 0.0702 - accuracy: 0.9784 - val_loss: 0.1086 - val_accuracy: 0.9726
Epoch  5/10 loss: 0.0472 - accuracy: 0.9856 - val_loss: 0.1197 - val_accuracy: 0.9738
Epoch  6/10 loss: 0.0328 - accuracy: 0.9896 - val_loss: 0.1195 - val_accuracy: 0.9758
Epoch  7/10 loss: 0.0208 - accuracy: 0.9929 - val_loss: 0.1094 - val_accuracy: 0.9776
Epoch  8/10 loss: 0.0112 - accuracy: 0.9962 - val_loss: 0.1135 - val_accuracy: 0.9794
Epoch  9/10 loss: 0.0051 - accuracy: 0.9986 - val_loss: 0.1074 - val_accuracy: 0.9800
Epoch 10/10 loss: 0.0027 - accuracy: 0.9995 - val_loss: 0.1088 - val_accuracy: 0.9794
loss: 0.0899 - accuracy: 0.9816
Final learning rate: 0.0001
```
#### Examples End:
