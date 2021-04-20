### Assignment: sequence_classification
#### Date: Deadline: May 03, 23:59
#### Points: 3 points
#### Examples: sequence_classification_examples

The goal of this assignment is to introduce recurrent neural networks.
Considering recurrent neural network, the assignment shows convergence speed and
illustrates exploding gradient issue. The network should process sequences of 50
small integers and compute parity for each prefix of the sequence. The inputs
are either 0/1, or vectors with one-hot representation of small integer.

Your goal is to modify the
[sequence_classification.py](https://github.com/ufal/npfl114/tree/master/labs/08/sequence_classification.py)
template and implement the following:
- Use specified RNN type (`SimpleRNN`, `GRU` and `LSTM`) and dimensionality.
- Process the sequence using the required RNN.
- Use additional hidden layer on the RNN outputs if requested.
- Implement gradient clipping if requested.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard. Concentrate on the way
how the RNNs converge, convergence speed, exploding gradient issues
and how gradient clipping helps:
- `--rnn_cell=SimpleRNN --sequence_dim=1`, `--rnn_cell=GRU --sequence_dim=1`, `--rnn_cell=LSTM --sequence_dim=1`
- the same as above but with `--sequence_dim=2`
- the same as above but with `--sequence_dim=10`
- `--rnn_cell=LSTM --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30` and the same with `--clip_gradient=1`
- the same as above but with `--rnn_cell=SimpleRNN`
- the same as above but with `--rnn_cell=GRU --hidden_layer=90`

#### Examples Start: sequence_classification_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 sequence_classification.py --rnn_cell SimpleRNN --epochs=5`
```
Epoch 1/5 loss: 0.7008 - accuracy: 0.5037 - val_loss: 0.6926 - val_accuracy: 0.5176
Epoch 2/5 loss: 0.6924 - accuracy: 0.5165 - val_loss: 0.6921 - val_accuracy: 0.5217
Epoch 3/5 loss: 0.6920 - accuracy: 0.5166 - val_loss: 0.6913 - val_accuracy: 0.5114
Epoch 4/5 loss: 0.6908 - accuracy: 0.5193 - val_loss: 0.6881 - val_accuracy: 0.5157
Epoch 5/5 loss: 0.6863 - accuracy: 0.5217 - val_loss: 0.6793 - val_accuracy: 0.5231
```
- `python3 sequence_classification.py --rnn_cell GRU --epochs=5`
```
Epoch 1/5 loss: 0.6930 - accuracy: 0.5109 - val_loss: 0.6917 - val_accuracy: 0.5157
Epoch 2/5 loss: 0.6905 - accuracy: 0.5170 - val_loss: 0.6823 - val_accuracy: 0.5143
Epoch 3/5 loss: 0.6342 - accuracy: 0.5925 - val_loss: 0.2222 - val_accuracy: 0.9695
Epoch 4/5 loss: 0.1759 - accuracy: 0.9760 - val_loss: 0.0930 - val_accuracy: 0.9882
Epoch 5/5 loss: 0.0754 - accuracy: 0.9938 - val_loss: 0.0381 - val_accuracy: 0.9986
```
- `python3 sequence_classification.py --rnn_cell LSTM --epochs=5`
```
Epoch 1/5 loss: 0.6931 - accuracy: 0.5131 - val_loss: 0.6927 - val_accuracy: 0.5153
Epoch 2/5 loss: 0.6924 - accuracy: 0.5158 - val_loss: 0.6902 - val_accuracy: 0.5156
Epoch 3/5 loss: 0.6874 - accuracy: 0.5174 - val_loss: 0.6748 - val_accuracy: 0.5285
Epoch 4/5 loss: 0.5799 - accuracy: 0.6247 - val_loss: 0.0695 - val_accuracy: 1.0000
Epoch 5/5 loss: 0.0482 - accuracy: 1.0000 - val_loss: 0.0183 - val_accuracy: 1.0000
```
- `python3 sequence_classification.py --rnn_cell LSTM --epochs=5 --hidden_layer=50`
```
Epoch 1/5 loss: 0.6884 - accuracy: 0.5129 - val_loss: 0.6614 - val_accuracy: 0.5309
Epoch 2/5 loss: 0.6544 - accuracy: 0.5362 - val_loss: 0.6378 - val_accuracy: 0.5301
Epoch 3/5 loss: 0.6319 - accuracy: 0.5482 - val_loss: 0.5836 - val_accuracy: 0.6181
Epoch 4/5 loss: 0.2933 - accuracy: 0.8366 - val_loss: 0.0030 - val_accuracy: 0.9998
Epoch 5/5 loss: 0.0023 - accuracy: 0.9999 - val_loss: 0.0010 - val_accuracy: 0.9999
```
- `python3 sequence_classification.py --rnn_cell LSTM --epochs=5 --hidden_layer=50 --clip_gradient=0.1`
```
Epoch 1/5 loss: 0.6884 - accuracy: 0.5130 - val_loss: 0.6615 - val_accuracy: 0.5302
Epoch 2/5 loss: 0.6544 - accuracy: 0.5364 - val_loss: 0.6373 - val_accuracy: 0.5293
Epoch 3/5 loss: 0.6304 - accuracy: 0.5517 - val_loss: 0.5875 - val_accuracy: 0.6107
Epoch 4/5 loss: 0.3835 - accuracy: 0.7753 - val_loss: 6.5897e-04 - val_accuracy: 1.0000
Epoch 5/5 loss: 0.0011 - accuracy: 0.9999 - val_loss: 1.6853e-04 - val_accuracy: 1.0000
```
#### Examples End:
