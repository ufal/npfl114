### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 15, 23:59
#### Points: 2 points
#### Examples: mnist_layers_activations_examples

Before solving the assignment, start by playing with
[example_keras_tensorboard.py](https://github.com/ufal/npfl114/tree/master/labs/01/example_keras_tensorboard.py),
in order to familiarize with TensorFlow and TensorBoard.
Run it, and when it finishes, run TensorBoard using `tensorboard --logdir logs`.
Then open <http://localhost:6006> in a browser and explore the active tabs.

**Your goal** is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl114/tree/master/labs/01/mnist_layers_activations.py)
template and implement the following:
- A number of hidden layers (including zero) can be specified on the command line
  using parameter `hidden_layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.
- Print the final accuracy on the test set.

#### Examples Start: mnist_layers_activations_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 mnist_layers_activations.py --hidden_layers=0 --activation=none`
```
Epoch  1/10 loss: 0.8272 - accuracy: 0.7869 - val_loss: 0.2755 - val_accuracy: 0.9308
Epoch  2/10 loss: 0.3328 - accuracy: 0.9089 - val_loss: 0.2419 - val_accuracy: 0.9342
Epoch  3/10 loss: 0.2995 - accuracy: 0.9165 - val_loss: 0.2269 - val_accuracy: 0.9392
Epoch  4/10 loss: 0.2886 - accuracy: 0.9197 - val_loss: 0.2219 - val_accuracy: 0.9414
Epoch  5/10 loss: 0.2778 - accuracy: 0.9222 - val_loss: 0.2202 - val_accuracy: 0.9430
Epoch  6/10 loss: 0.2745 - accuracy: 0.9234 - val_loss: 0.2171 - val_accuracy: 0.9416
Epoch  7/10 loss: 0.2669 - accuracy: 0.9246 - val_loss: 0.2152 - val_accuracy: 0.9420
Epoch  8/10 loss: 0.2615 - accuracy: 0.9263 - val_loss: 0.2159 - val_accuracy: 0.9424
Epoch  9/10 loss: 0.2561 - accuracy: 0.9280 - val_loss: 0.2156 - val_accuracy: 0.9404
Epoch 10/10 loss: 0.2596 - accuracy: 0.9270 - val_loss: 0.2146 - val_accuracy: 0.9434
loss: 0.2637 - accuracy: 0.9259
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=none`
```
Epoch  1/10 loss: 0.5384 - accuracy: 0.8430 - val_loss: 0.2438 - val_accuracy: 0.9350
Epoch  2/10 loss: 0.2951 - accuracy: 0.9166 - val_loss: 0.2332 - val_accuracy: 0.9350
Epoch  3/10 loss: 0.2816 - accuracy: 0.9217 - val_loss: 0.2359 - val_accuracy: 0.9306
Epoch  4/10 loss: 0.2808 - accuracy: 0.9225 - val_loss: 0.2283 - val_accuracy: 0.9384
Epoch  5/10 loss: 0.2705 - accuracy: 0.9227 - val_loss: 0.2341 - val_accuracy: 0.9370
Epoch  6/10 loss: 0.2718 - accuracy: 0.9234 - val_loss: 0.2333 - val_accuracy: 0.9388
Epoch  7/10 loss: 0.2669 - accuracy: 0.9253 - val_loss: 0.2223 - val_accuracy: 0.9412
Epoch  8/10 loss: 0.2595 - accuracy: 0.9281 - val_loss: 0.2471 - val_accuracy: 0.9342
Epoch  9/10 loss: 0.2573 - accuracy: 0.9270 - val_loss: 0.2293 - val_accuracy: 0.9368
Epoch 10/10 loss: 0.2615 - accuracy: 0.9264 - val_loss: 0.2318 - val_accuracy: 0.9400
loss: 0.2795 - accuracy: 0.9241
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=relu`
```
Epoch  1/10 loss: 0.5379 - accuracy: 0.8500 - val_loss: 0.1459 - val_accuracy: 0.9612
Epoch  2/10 loss: 0.1563 - accuracy: 0.9553 - val_loss: 0.1128 - val_accuracy: 0.9682
Epoch  3/10 loss: 0.1052 - accuracy: 0.9697 - val_loss: 0.0966 - val_accuracy: 0.9714
Epoch  4/10 loss: 0.0792 - accuracy: 0.9765 - val_loss: 0.0864 - val_accuracy: 0.9744
Epoch  5/10 loss: 0.0627 - accuracy: 0.9814 - val_loss: 0.0818 - val_accuracy: 0.9768
Epoch  6/10 loss: 0.0500 - accuracy: 0.9857 - val_loss: 0.0829 - val_accuracy: 0.9772
Epoch  7/10 loss: 0.0394 - accuracy: 0.9881 - val_loss: 0.0747 - val_accuracy: 0.9792
Epoch  8/10 loss: 0.0328 - accuracy: 0.9905 - val_loss: 0.0746 - val_accuracy: 0.9788
Epoch  9/10 loss: 0.0239 - accuracy: 0.9934 - val_loss: 0.0845 - val_accuracy: 0.9762
Epoch 10/10 loss: 0.0231 - accuracy: 0.9936 - val_loss: 0.0806 - val_accuracy: 0.9778
loss: 0.0829 - accuracy: 0.9773
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=tanh`
```
Epoch  1/10 loss: 0.5338 - accuracy: 0.8483 - val_loss: 0.1668 - val_accuracy: 0.9570
Epoch  2/10 loss: 0.1855 - accuracy: 0.9478 - val_loss: 0.1262 - val_accuracy: 0.9648
Epoch  3/10 loss: 0.1271 - accuracy: 0.9640 - val_loss: 0.1001 - val_accuracy: 0.9724
Epoch  4/10 loss: 0.0966 - accuracy: 0.9716 - val_loss: 0.0918 - val_accuracy: 0.9738
Epoch  5/10 loss: 0.0742 - accuracy: 0.9784 - val_loss: 0.0813 - val_accuracy: 0.9774
Epoch  6/10 loss: 0.0605 - accuracy: 0.9832 - val_loss: 0.0811 - val_accuracy: 0.9750
Epoch  7/10 loss: 0.0471 - accuracy: 0.9872 - val_loss: 0.0759 - val_accuracy: 0.9774
Epoch  8/10 loss: 0.0385 - accuracy: 0.9902 - val_loss: 0.0761 - val_accuracy: 0.9762
Epoch  9/10 loss: 0.0298 - accuracy: 0.9929 - val_loss: 0.0783 - val_accuracy: 0.9766
Epoch 10/10 loss: 0.0257 - accuracy: 0.9945 - val_loss: 0.0788 - val_accuracy: 0.9744
loss: 0.0822 - accuracy: 0.9751
```
- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=sigmoid`
```
Epoch  1/10 loss: 0.8219 - accuracy: 0.7952 - val_loss: 0.2150 - val_accuracy: 0.9400
Epoch  2/10 loss: 0.2485 - accuracy: 0.9301 - val_loss: 0.1632 - val_accuracy: 0.9562
Epoch  3/10 loss: 0.1864 - accuracy: 0.9477 - val_loss: 0.1322 - val_accuracy: 0.9636
Epoch  4/10 loss: 0.1513 - accuracy: 0.9560 - val_loss: 0.1163 - val_accuracy: 0.9676
Epoch  5/10 loss: 0.1235 - accuracy: 0.9646 - val_loss: 0.1041 - val_accuracy: 0.9718
Epoch  6/10 loss: 0.1069 - accuracy: 0.9702 - val_loss: 0.0957 - val_accuracy: 0.9722
Epoch  7/10 loss: 0.0889 - accuracy: 0.9746 - val_loss: 0.0887 - val_accuracy: 0.9746
Epoch  8/10 loss: 0.0774 - accuracy: 0.9785 - val_loss: 0.0869 - val_accuracy: 0.9756
Epoch  9/10 loss: 0.0641 - accuracy: 0.9832 - val_loss: 0.0845 - val_accuracy: 0.9760
Epoch 10/10 loss: 0.0594 - accuracy: 0.9842 - val_loss: 0.0805 - val_accuracy: 0.9772
loss: 0.0862 - accuracy: 0.9741
```
- `python3 mnist_layers_activations.py --hidden_layers=3 --activation=relu`
```
Epoch  1/10 loss: 0.4989 - accuracy: 0.8471 - val_loss: 0.1121 - val_accuracy: 0.9688
Epoch  2/10 loss: 0.1168 - accuracy: 0.9645 - val_loss: 0.1028 - val_accuracy: 0.9692
Epoch  3/10 loss: 0.0784 - accuracy: 0.9756 - val_loss: 0.1176 - val_accuracy: 0.9654
Epoch  4/10 loss: 0.0586 - accuracy: 0.9810 - val_loss: 0.0860 - val_accuracy: 0.9732
Epoch  5/10 loss: 0.0451 - accuracy: 0.9849 - val_loss: 0.0867 - val_accuracy: 0.9778
Epoch  6/10 loss: 0.0398 - accuracy: 0.9869 - val_loss: 0.0884 - val_accuracy: 0.9782
Epoch  7/10 loss: 0.0303 - accuracy: 0.9898 - val_loss: 0.0797 - val_accuracy: 0.9818
Epoch  8/10 loss: 0.0256 - accuracy: 0.9917 - val_loss: 0.0892 - val_accuracy: 0.9796
Epoch  9/10 loss: 0.0218 - accuracy: 0.9930 - val_loss: 0.1074 - val_accuracy: 0.9732
Epoch 10/10 loss: 0.0220 - accuracy: 0.9927 - val_loss: 0.0821 - val_accuracy: 0.9796
loss: 0.0883 - accuracy: 0.9779
```
- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=relu`
```
Epoch  1/10 loss: 0.6597 - accuracy: 0.7806 - val_loss: 0.1348 - val_accuracy: 0.9622
Epoch  2/10 loss: 0.1533 - accuracy: 0.9561 - val_loss: 0.1172 - val_accuracy: 0.9670
Epoch  3/10 loss: 0.1154 - accuracy: 0.9680 - val_loss: 0.0991 - val_accuracy: 0.9708
Epoch  4/10 loss: 0.0912 - accuracy: 0.9737 - val_loss: 0.1112 - val_accuracy: 0.9704
Epoch  5/10 loss: 0.0758 - accuracy: 0.9795 - val_loss: 0.1060 - val_accuracy: 0.9732
Epoch  6/10 loss: 0.0729 - accuracy: 0.9794 - val_loss: 0.1077 - val_accuracy: 0.9730
Epoch  7/10 loss: 0.0647 - accuracy: 0.9825 - val_loss: 0.0921 - val_accuracy: 0.9734
Epoch  8/10 loss: 0.0554 - accuracy: 0.9845 - val_loss: 0.0994 - val_accuracy: 0.9756
Epoch  9/10 loss: 0.0503 - accuracy: 0.9871 - val_loss: 0.1114 - val_accuracy: 0.9720
Epoch 10/10 loss: 0.0470 - accuracy: 0.9875 - val_loss: 0.1084 - val_accuracy: 0.9740
loss: 0.1119 - accuracy: 0.9736
```
- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=sigmoid`
```
Epoch  1/10 loss: 2.3115 - accuracy: 0.1026 - val_loss: 1.8614 - val_accuracy: 0.2174
Epoch  2/10 loss: 1.8910 - accuracy: 0.1963 - val_loss: 1.8708 - val_accuracy: 0.2064
Epoch  3/10 loss: 1.8796 - accuracy: 0.1998 - val_loss: 1.8007 - val_accuracy: 0.2030
Epoch  4/10 loss: 1.8249 - accuracy: 0.2047 - val_loss: 1.4527 - val_accuracy: 0.3074
Epoch  5/10 loss: 1.2759 - accuracy: 0.4293 - val_loss: 0.8859 - val_accuracy: 0.6154
Epoch  6/10 loss: 0.9357 - accuracy: 0.5910 - val_loss: 0.8584 - val_accuracy: 0.6884
Epoch  7/10 loss: 0.8281 - accuracy: 0.6777 - val_loss: 0.6917 - val_accuracy: 0.7296
Epoch  8/10 loss: 0.7334 - accuracy: 0.7111 - val_loss: 0.6801 - val_accuracy: 0.7124
Epoch  9/10 loss: 0.7111 - accuracy: 0.7132 - val_loss: 0.7223 - val_accuracy: 0.6916
Epoch 10/10 loss: 0.6875 - accuracy: 0.7243 - val_loss: 0.6183 - val_accuracy: 0.7850
loss: 0.6737 - accuracy: 0.7623
```
#### Examples End:
